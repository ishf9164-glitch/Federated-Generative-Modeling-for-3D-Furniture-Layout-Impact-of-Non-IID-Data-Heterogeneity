# ---- Base image: PyTorch + CUDA runtime (GPU node 用) ----
    FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

    # ---- System deps ----
    RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        libgl1 libglib2.0-0 \
     && rm -rf /var/lib/apt/lists/*
    
    # ---- Python deps ----
    RUN pip install --no-cache-dir -U pip \
     && pip install --no-cache-dir \
        numpy tqdm pyyaml pillow trimesh wandb
    
    # ---- Copy project ----
    WORKDIR /workspace
    COPY . /workspace/diverse_synth
    
    # ---- Create output dirs used by configs (optional but helpful) ----
    RUN mkdir -p /workspace/diverse_synth/savepoint /workspace/diverse_synth/output
    
    # ---- Runtime env ----
    ENV PYTHONUNBUFFERED=1 \
        PYTHONDONTWRITEBYTECODE=1
    
    # IMPORTANT:
    # Your training commands use relative paths like ../config/bedroom_config.yaml
    # so we set the working directory to scripts/
    WORKDIR /workspace/diverse_synth/scripts
    
    # Default: open a shell so you can run the exact commands
    CMD ["python", "/workspace/diverse_synth/client.py"]
    