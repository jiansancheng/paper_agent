import os
# 设置镜像，防止代码里没读到环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

print("🚀 开始下载 CLIP 模型...")
snapshot_download(
    repo_id="OFA-Sys/chinese-clip-vit-base-patch16",#sentence-transformers/clip-ViT-B-32-multilingual-v1
    local_dir="./models/chinese-clip-vit-base-patch16",
    local_dir_use_symlinks=False,  # 关键：确保下载的是真实文件而不是软链接
    resume_download=True           # 这里可以用 python 参数控制
)
print("✅ clip下载完成！")

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="/data0/jycheng/homework/paper_agent/models/bge-m3",
    ignore_patterns=[
        "imgs/**",
        ".DS_Store",
        "*.onnx",
        "model.onnx_data",
        "LICENSE",
        "README.md",
        "*.png",
        "*.jpg",
        "*.md"
    ]
)
print("✅ bge下载完成！")