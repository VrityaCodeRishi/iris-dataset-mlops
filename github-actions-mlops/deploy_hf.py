import os
import json
from huggingface_hub import HfApi, login

login(token=os.getenv("HUGGINGFACE_TOKEN"))

repo_id = os.environ["HUGGINGFACE_REPO_ID"]

api = HfApi()

api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id=repo_id,
    repo_type="model",
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
