#!/usr/bin/env bash
set -e

ENV_NAME=feddistill310

conda create -n ${ENV_NAME} python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

python -m pip install numpy==1.26.4 scipy matplotlib pillow tqdm wandb torchmetrics

python - <<'PY'
import torch, torchvision, wandb, torchmetrics, tqdm, numpy, PIL, matplotlib
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("wandb:", wandb.__version__)
print("torchmetrics:", torchmetrics.__version__)
print("FedDistill deps ok")
PY