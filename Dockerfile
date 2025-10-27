FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# install Python and venv + build deps
RUN apt-get update && apt-get install -y \
  python3 python3-venv python3-pip build-essential git wget ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Ensure `python` points to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# create virtualenv, activate it and install requirements into it
RUN python3 -m venv /opt/venv \
  && /opt/venv/bin/pip install --upgrade pip \
  && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# make the venv first in PATH for subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

COPY app/api/app.py /app/app.py


ENV MODEL_PATH=jimjunior/event-diffusion-model
EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
