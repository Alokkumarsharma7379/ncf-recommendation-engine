from pathlib import Path

import requests
from tqdm import tqdm

CATEGORY = "Office_Products"
DATASET_URL = (
    f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    f"resolve/main/raw/review_categories/{CATEGORY}.jsonl"
)
SENTINEL = f"{CATEGORY}.jsonl"


def download(target_dir: str = "data/raw") -> Path:
    root = Path(target_dir)
    out_path = root / SENTINEL

    if out_path.exists():
        print(f"Dataset already present at {out_path!s}, skipping download.")
        return root

    root.mkdir(parents=True, exist_ok=True)

    response = requests.get(DATASET_URL, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=CATEGORY
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Dataset ready at {out_path!s}")
    return root


if __name__ == "__main__":
    download()