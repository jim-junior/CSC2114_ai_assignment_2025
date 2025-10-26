import argparse
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# --- 1. CONFIGURATION (Your Established Parameters) ---
# NOTE: These values are hardcoded based on our previous discussion.
# You can change these values here before running the script.
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
# Should contain images and a metadata.csv
TRAIN_DATA_DIR = "/content/my_uganda_speaker_images"
OUTPUT_DIR = "/output"
CAPTION_COLUMN = "caption"
MAX_TRAIN_STEPS = 1000
LEARNING_RATE = 1e-4
LORA_RANK = 64
RESOLUTION = 512
BATCH_SIZE = 1  # Per device batch size

# --- 2. CUSTOM DATASET CLASS (Mimics your CSV/Image structure) ---


class CaptionDataset(Dataset):
    """
    A custom PyTorch Dataset class to load images and captions based on a metadata CSV.
    The CSV is expected to be in the TRAIN_DATA_DIR.
    """

    def __init__(self, data_root, tokenizer, resolution=512):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.metadata_path = os.path.join(data_root, "metadata.csv")

        if not os.path.exists(self.metadata_path):
            # Create a placeholder CSV if it doesn't exist for testing,
            # but warn the user they must replace it with real data.
            print("WARNING: metadata.csv not found. Using placeholder data!")
            self.metadata = pd.DataFrame({
                'file_name': [f"placeholder_image_{i}.png" for i in range(10)],
                CAPTION_COLUMN: [
                    f"a photo of a speaker giving a talk at Pycon Uganda, tag{i}" for i in range(10)]
            })
            self.image_files = []  # Will be empty if real images aren't present
        else:
            self.metadata = pd.read_csv(self.metadata_path)
            self.image_files = [os.path.join(
                self.data_root, f) for f in self.metadata['file_name'].tolist()]

        self.num_samples = len(self.metadata)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Get Caption
        caption = self.metadata.iloc[idx][CAPTION_COLUMN]
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # 2. Get Image (Placeholder or Actual)
        if len(self.image_files) == 0:
            # If no real images, return a black placeholder tensor
            image = torch.zeros(
                (3, self.resolution, self.resolution), dtype=torch.float32)
        else:
            try:
                # Load actual image file
                image_path = self.image_files[idx]
                image = Image.open(image_path).convert(
                    "RGB").resize((self.resolution, self.resolution))
                image = torch.tensor(image).permute(
                    2, 0, 1).float() / 127.5 - 1.0  # Normalize to [-1, 1]
            except Exception as e:
                # Handle missing/corrupt image, use a black tensor and log the issue
                print(
                    f"Error loading image {self.metadata.iloc[idx]['file_name']}: {e}")
                image = torch.zeros(
                    (3, self.resolution, self.resolution), dtype=torch.float32)

        return {"pixel_values": image, "input_ids": input_ids}


# --- 3. TRAINING FUNCTION ---
def train_lora():
    set_seed(42)

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,  # Matches the accumulation in our command
        mixed_precision="fp16",
        log_with="tensorboard",
        project_dir=os.path.join(OUTPUT_DIR, "logs"),
    )

    # 1. Load Pre-trained Components
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")

    # Disable VAE and Text Encoder training
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 2. LoRA Setup on UNet (The Core of LoRA Fine-Tuning)
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,  # Set alpha equal to rank for standard scaling
        init_lora_weights="gaussian",
        # Standard attention layers in UNet
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Apply LoRA to the UNet (this adds the trainable LoRA adapters)
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    # Note: Only the LoRA adapters are now trainable, not the base UNet weights.

    # 3. Setup Optimizer, Scheduler, and Noise
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Denoising Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        MODEL_NAME, subfolder="scheduler")

    # 4. Data Loading
    train_dataset = CaptionDataset(
        data_root=TRAIN_DATA_DIR,
        tokenizer=tokenizer,
        resolution=RESOLUTION
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count() or 0,
        pin_memory=True,
    )

    # Setup Scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100 * accelerator.num_processes,
        num_training_steps=MAX_TRAIN_STEPS * accelerator.num_processes,
    )

    # 5. Prepare with Accelerator (Crucial step for multi-GPU/mixed precision)
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # 6. Training Loop
    total_batch_size = BATCH_SIZE * accelerator.num_processes * \
        accelerator.gradient_accumulation_steps
    accelerator.print(f"***** Running LoRA Training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(
        f"  Total batch size (w/ accumulation) = {total_batch_size}")
    accelerator.print(f"  Total training steps = {MAX_TRAIN_STEPS}")

    # Start training
    global_step = 0
    progress_bar = tqdm(
        range(MAX_TRAIN_STEPS),
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    for step, batch in enumerate(train_dataloader):
        if global_step >= MAX_TRAIN_STEPS:
            break

        with accelerator.accumulate(unet):
            # Encode text (get text embeddings)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # VAE encode the images to latent space
            latents = vae.encode(batch["pixel_values"].to(
                unet.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise to add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

            # Add noise to the latents according to the noise magnitude at each timestep (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(
                latents, noise, timesteps)

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps,
                              encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Calculate loss
            loss = F.mse_loss(model_pred.float(),
                              target.float(), reduction="mean")

            # Backpropagate and update weights
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"train_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

        progress_bar.set_postfix({"loss": loss.detach().item()})

    # 7. Save the trained LoRA weights
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)

    # Save only the UNet LoRA weights in the `.safetensors` format
    lora_weights_path = os.path.join(
        OUTPUT_DIR, "pytorch_lora_weights.safetensors")
    unwrapped_unet.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    accelerator.print(
        f"LoRA weights saved successfully to {lora_weights_path}")
    accelerator.end_training()


if __name__ == "__main__":
    train_lora()
