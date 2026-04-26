import os
# 强制走国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

repo_id = "0xaryan/music-classifier"
save_dir = "D:\File_Python\digital_signal_process\datasets_download" # 指定保存原始文件的目录

print("开始下载原始音频文件和CSV表格...")

# 将整个仓库的文件原封不动下载到本地
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=save_dir,
    local_dir_use_symlinks=False, # 确保下载的是实体文件
    resume_download=True,         # 开启断点续传
    max_workers=1                 # 单线程稳定下载
)

print(f"下载成功！你可以直接进入 {save_dir}/genres_original 文件夹查看和切割音频了。")
