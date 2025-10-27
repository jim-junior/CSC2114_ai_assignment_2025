from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import uuid
import os
import base64
from io import BytesIO
from PIL import Image

MODEL_PATH = "jimjunior/event-diffusion-model"

app = FastAPI(title="Event Gen API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ✅ allow all origins
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],       # allow all HTTP methods
    allow_headers=["*"],       # allow all headers
)

# Load pipeline once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = None


@app.on_event("startup")
def load_model():
    global pipe
    # Use `torch_dtype=torch.float16` if GPU supports it
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # consider hooking moderation instead
    )
    pipe.to(device)
    # Optionally enable attention slicing
    pipe.enable_attention_slicing()
    # Optionally enable xformers (if installed)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass


class GenRequest(BaseModel):
    prompt: str
    steps: int = 25
    guidance_scale: float = 7.5
    seed: int | None = None
    width: int = 512
    height: int = 512


@app.post("/generate")
def generate(req: GenRequest):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    img = pipe(
        req.prompt,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale
    ).images[0]

    # encode to base64 and return (or save to disk / S3 and return URL)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": b64, "size": f"{req.width}x{req.height}"}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Event Gen API. Use the /generate endpoint to create images."}
