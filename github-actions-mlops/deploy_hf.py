import os
from huggingface_hub import HfApi, login

hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set or is empty")


login(token=hf_token)


repo_id = os.environ.get("HF_REPO_ID", "your-username/iris-classifier")
if repo_id == "your-username/iris-classifier":
    print("Warning: Using default repo ID. Set HF_REPO_ID environment variable.")

api = HfApi()
api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id=repo_id,
    repo_type="model",
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
