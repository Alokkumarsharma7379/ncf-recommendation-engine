import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
SENTINEL = "ml-100k/u.data"


def download(target_dir: str = "data/raw") -> Path:
    root = Path(target_dir)
    if (root / SENTINEL).exists():
        print(f"Dataset already present at {root / SENTINEL!s}, skipping download.")
        return root

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "ml-100k.zip"

    response = requests.get(DATASET_URL, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="ml-100k"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    zip_path.unlink()
    print(f"Dataset ready at {root / 'ml-100k'!s}")
    return root


if __name__ == "__main__":
    download()