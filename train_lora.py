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
TRAIN_DATA_DIR = "/content/drive/MyDrive/AI_DATASET"
METADATA_FILE_PATH = "/content/scripts/metadata.csv"
OUTPUT_DIR = "/content/output"
CAPTION_COLUMN = "caption"
MAX_TRAIN_STEPS = 10
LEARNING_RATE = 1e-4
LORA_RANK = 64
RESOLUTION = 512
BATCH_SIZE = 1  # Per device batch size

# --- 2. CUSTOM DATASET CLASS (Mimics your CSV/Image structure) ---


class CaptionDataset(Dataset):
    """
    Loads metadata from <data_root>/metadata.csv and filters out entries
    whose image files do not exist on disk. Returns dicts with:
      - "pixel_values": tensor shape (3, resolution, resolution), values in [-1, 1]
      - "input_ids": tokenized caption (torch.LongTensor)

    Expects CSV columns:
      - 'name' : filename of the image (will be joined with data_root)
      - CAPTION_COLUMN : caption text

    If metadata.csv is missing, a placeholder small dataset is created.
    """

    def __init__(self, data_root, tokenizer, resolution=512):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.resolution = resolution

        # Prefer metadata.csv inside the data_root (matches your images).
        self.metadata_path = METADATA_FILE_PATH

        if not os.path.exists(self.metadata_path):
            print("WARNING: metadata.csv not found in data_root. Using placeholder data!")
            self.metadata = pd.DataFrame({
                "name": [f"placeholder_image_{i}.png" for i in range(10)],
                CAPTION_COLUMN: [
                    f"a photo of a speaker giving a talk at Pycon Uganda, tag{i}" for i in range(10)
                ]
            })
            self.image_files = []  # no real images
        else:
            raw_meta = pd.read_csv(self.metadata_path)

            # Clean/normalize filenames (strip whitespace)
            if "name" not in raw_meta.columns:
                raise ValueError(
                    "metadata.csv must contain a 'name' column with image filenames.")
            raw_meta["name"] = raw_meta["name"].astype(str).str.strip()

            # Build the absolute image paths
            raw_meta["image_path"] = raw_meta["name"].apply(
                lambda n: os.path.join(self.data_root, n))

            # Check which files exist
            raw_meta["exists"] = raw_meta["image_path"].apply(
                lambda p: os.path.exists(p))

            num_total = len(raw_meta)
            num_exists = int(raw_meta["exists"].sum())
            num_missing = num_total - num_exists

            if num_missing > 0:
                print(
                    f"Found {num_total} entries in metadata.csv — skipping {num_missing} missing files, keeping {num_exists} entries.")

            # Keep only rows with existing files
            filtered_meta = raw_meta[raw_meta["exists"]].reset_index(drop=True)

            # If after filtering we end up with zero images, warn but keep dataframe
            if len(filtered_meta) == 0:
                print(
                    "WARNING: No valid image files found in data_root. Dataset will return black placeholder tensors.")

            # Save metadata and image file list
            self.metadata = filtered_meta
            self.image_files = filtered_meta["image_path"].tolist()

        self.num_samples = len(self.metadata)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Safeguard if dataset is empty
        if self.num_samples == 0:
            caption = ""
        else:
            caption = self.metadata.iloc[idx].get(CAPTION_COLUMN, "")
            if pd.isna(caption):
                caption = ""

        # Tokenize caption (ensure we return 1D tensor)
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Load image (or placeholder)
        if len(self.image_files) == 0:
            # Return black placeholder
            image = torch.zeros(
                (3, self.resolution, self.resolution), dtype=torch.float32)
        else:
            image_path = self.image_files[idx]
            try:
                img = Image.open(image_path).convert("RGB").resize(
                    (self.resolution, self.resolution))
                # Convert via numpy to avoid PIL dtype inference problems, then to float tensor
                import numpy as np
                # shape (H, W, C), dtype float32 in [0,255]
                arr = np.array(img).astype(np.float32)
                # Convert to CHW and normalize to [-1, 1]
                arr = arr.transpose(2, 0, 1)  # C,H,W
                tensor = torch.from_numpy(arr) / 127.5 - 1.0
                image = tensor.to(dtype=torch.float32)
            except Exception as e:
                # If anything goes wrong (corrupt image etc.), log and return black tensor
                print(f"Error loading image {image_path}: {e}")
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
    unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler
    )

    # --- FIX 1: Move non-trainable components to the GPU ---
    # The VAE and Text Encoder are not passed to accelerator.prepare, so they
    # must be explicitly moved to the device determined by the Accelerator.
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    # --------------------------------------------------------

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
            # --- FIX 2: Move input_ids to the GPU before encoding ---
            # Input IDs are on CPU from the DataLoader; move them to GPU
            # where the text_encoder is now located.
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(accelerator.device)

            # Now safe to call text_encoder
            # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            # input_ids = batch["input_ids"].to(accelerator.device)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            # --------------------------------------------------------

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
