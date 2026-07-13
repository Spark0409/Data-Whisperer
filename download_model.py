import os
# 设置镜像（必须在 import huggingface_hub 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen2.5-3B-Instruct"
local_dir = "D:/models/Qwen2.5-3B-Instruct"

os.makedirs(local_dir, exist_ok=True)

print(f"Downloading {model_name} from mirror to {local_dir}...")
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
)
print(f"Download complete! Model saved to {local_dir}")
