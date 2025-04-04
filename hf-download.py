from huggingface_hub import snapshot_download
import subprocess

# export HF_ENDPOINT=https://hf-mirror.com
# subprocess.check_output(["export", "HF_ENDPOINT=https://hf-mirror.com"])
# meta-llama/Llama-3.2-1B-Instruct
# microsoft/Phi-3-medium-4k-instruct
# lmsys/vicuna-7b-v1.5
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
repo_id = "meta-llama/Llama-3.2-1B-Instruct"

local_dir = "downloads/" + repo_id.split("/")[1]
snapshot_download(repo_id=repo_id, local_dir=local_dir)