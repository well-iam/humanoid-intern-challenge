"""Upload the staged dataset folder (scripts/hf_stage.py) to a Hugging Face dataset repo.

Needs a write token: `hf auth login` once, or HF_TOKEN set. Creates the repo if it does not exist (public unless
--private), fills the repo id into the dataset card, then uploads one top-level part per commit (the card, then
clips/, then pusht/), so a broken connection costs one part. Files already on the Hub are skipped, so rerunning
resumes.

    uv run python scripts/hf_upload.py --repo-id <user>/pushing-world-model --folder C:/Users/william.notaro/dev/pwm-data/hf-staging
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

PLACEHOLDER = "<HF_REPO_ID>"  # in the dataset card (docs/hf-dataset-card.md)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--repo-id", required=True, help="e.g. <user>/pushing-world-model")
parser.add_argument("--folder", required=True, help="the staging folder")
parser.add_argument("--private", action="store_true", help="create the repo private")
args = parser.parse_args()
folder = Path(args.folder)

card = folder / "README.md"
card.write_text(card.read_text(encoding="utf-8").replace(PLACEHOLDER, args.repo_id), encoding="utf-8")

api = HfApi()
api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
for part, patterns in [("dataset card", ["README.md"]), ("my clips", ["clips/**"]), ("Push-T results", ["pusht/**"])]:
    print(f"uploading {part} ...")
    api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=folder, allow_patterns=patterns,
                      commit_message=f"Add {part}")
print(f"https://huggingface.co/datasets/{args.repo_id}")
