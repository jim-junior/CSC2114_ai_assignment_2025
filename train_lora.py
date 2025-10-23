import os
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = orig_linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        out, inp = self.linear.out_features, self.linear.in_features
        self.r = r
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(out, r) * 1e-3)
        self.B = nn.Parameter(torch.randn(inp, r) * 1e-3)
        self.scaling = alpha / max(1, r)

    def forward(self, x: torch.Tensor):
        base = F.linear(x, self.linear.weight, self.linear.bias)
        deltaW = (self.A @ self.B.T)
        lora_out = F.linear(x, deltaW) * self.scaling
        return base + lora_out

    def merge_to_linear(self):
        deltaW = (self.A @ self.B.T) * self.scaling
        self.linear.weight.data += deltaW.to(self.linear.weight.device)
        self.A.requires_grad = False
        self.B.requires_grad = False


class EventDataset(Dataset):
    def __init__(self, images_dir, captions_json, resolution=512, tokenizer=None, max_length=77):
        self.images_dir = Path(images_dir)
        with open(captions_json, "r") as f:
            captions = json.load(f)
        items = []
        for fname, cap in captions.items():
            p = self.images_dir / fname
            if p.exists():
                items.append((str(p), cap))
        self.items = items
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, cap = self.items[idx]
        img = Image.open(path).convert("RGB")
        pv = self.transform(img)
        if self.tokenizer:
            toks = self.tokenizer(cap, padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
            return {"pixel_values": pv, "caption": cap, "input_ids": toks.input_ids[0], "attention_mask": toks.attention_mask[0]}
        return {"pixel_values": pv, "caption": cap, "input_ids": None, "attention_mask": None}


def apply_lora_to_unet(unet_model, r=4, alpha=1.0, verbose=True):
    replaced = []
    for name, module in unet_model.named_modules():
        if isinstance(module, nn.Linear):
            if any(k in name.lower() for k in ("to_q", "to_k", "to_v", "proj_attn", "to_out", "attn")):
                parent_path = name.split(".")[:-1]
                attr = name.split(".")[-1]
                parent = unet_model
                ok = True
                try:
                    for p in parent_path:
                        parent = getattr(parent, p)
                except Exception:
                    ok = False
                if not ok:
                    continue
                orig = getattr(parent, attr)
                lora_layer = LoRALinear(orig, r=r, alpha=alpha)
                setattr(parent, attr, lora_layer)
                replaced.append(name)
    if verbose:
        print(
            f"[apply_lora_to_unet] Replaced {len(replaced)} linear modules (examples):")
        for ex in replaced[:10]:
            print("  -", ex)
    return replaced


def encode_images_to_latents(vae, pixel_values, device):
    with torch.no_grad():
        latent_dist = vae.encode(pixel_values.to(device)).latent_dist
        latents = latent_dist.sample() * 0.18215
    return latents


def get_text_embeddings(tokenizer, text_encoder, captions, device):
    toks = tokenizer(captions, padding="max_length",
                     truncation=True, max_length=77, return_tensors="pt")
    input_ids = toks.input_ids.to(device)
    attn_mask = toks.attention_mask.to(device)
    with torch.no_grad():
        emb = text_encoder(input_ids=input_ids, attention_mask=attn_mask)[0]
    return emb


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14").to(device)

    print("Loading VAE and UNet...")
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet").to(device)

    for p in vae.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = False

    print("Applying LoRA to UNet...")
    apply_lora_to_unet(unet, r=args.lora_rank, alpha=args.lora_alpha)

    dataset = EventDataset(args.images_dir, args.captions_json,
                           resolution=args.resolution, tokenizer=tokenizer)
    print("Dataset size:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    lora_params = [p for n, p in unet.named_parameters() if p.requires_grad]
    print("Trainable LoRA params count:", sum(p.numel() for p in lora_params))
    opt = torch.optim.AdamW(lora_params, lr=args.learning_rate)

    os.makedirs(args.output_dir, exist_ok=True)
    unet.train()
    vae.eval()
    text_encoder.eval()

    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"]
            captions = batch["caption"]
            latents = encode_images_to_latents(vae, pixel_values, device)
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(
                latents, noise, timesteps)
            text_embeds = get_text_embeddings(
                tokenizer, text_encoder, captions, device)

            opt.zero_grad()
            model_out = unet(noisy_latents, timesteps,
                             encoder_hidden_states=text_embeds)
            noise_pred = model_out.sample if hasattr(
                model_out, "sample") else model_out
            loss = F.mse_loss(noise_pred, noise.to(device))
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"Step {global_step}/{args.max_train_steps} loss={loss.item():.6f}")

            if global_step % args.save_every == 0 or global_step >= args.max_train_steps:
                lora_state = {}
                for name, module in unet.named_modules():
                    if isinstance(module, LoRALinear):
                        lora_state[name + ".A"] = module.A.detach().cpu()
                        lora_state[name + ".B"] = module.B.detach().cpu()
                        lora_state[name +
                                   ".alpha"] = torch.tensor(module.alpha)
                        lora_state[name + ".r"] = torch.tensor(module.r)
                ckpt_path = os.path.join(
                    args.output_dir, f"lora_{global_step}.pt")
                torch.save(lora_state, ckpt_path)
                print("Saved LoRA checkpoint ->", ckpt_path)

            if global_step >= args.max_train_steps:
                break

    if args.merge_lora:
        print("Merging LoRA into UNet weights (in-place).")
        for name, module in unet.named_modules():
            if isinstance(module, LoRALinear):
                module.merge_to_linear()
        merged_dir = os.path.join(args.output_dir, "merged_unet")
        os.makedirs(merged_dir, exist_ok=True)
        unet.save_pretrained(merged_dir)
        print("Merged UNet saved to", merged_dir)

    print("Training complete. Final step:", global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--captions_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/lora_event")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--merge_lora", action="store_true")
    args = parser.parse_args()
    main(args)
