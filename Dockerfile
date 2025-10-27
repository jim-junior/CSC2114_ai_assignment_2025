FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

# Basic deps
RUN apt-get update && apt-get install -y python3-pip git wget build-essential

# Create app dir
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/api/app.py /app/app.py


ENV MODEL_PATH=jimjunior/event-diffusion-model
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
