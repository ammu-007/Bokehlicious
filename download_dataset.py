"""
Download the RealBokeh_3MP dataset from HuggingFace in Parquet format.

The parquet files are a partial snapshot (~5 GB train, much smaller than raw images).
Each parquet file contains embedded JPEG bytes for source and target images,
plus metadata (target_av, scene_id, etc.).

Usage:
    # Download partial-train + partial-validation (recommended for training):
    .venv\\Scripts\\python.exe download_dataset.py --split train
    .venv\\Scripts\\python.exe download_dataset.py --split validation
    .venv\\Scripts\\python.exe download_dataset.py --split test

    # Download everything:
    .venv\\Scripts\\python.exe download_dataset.py --split all
"""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "timseizinger/RealBokeh_3MP"
PARQUET_BRANCH = "refs/convert/parquet"
LOCAL_DIR = "./dataset/RealBokeh_Parquet"

SPLIT_MAP = {
    "train":      "default/partial-train",
    "validation": "default/partial-validation",
    "test":       "default/partial-test",
}


def download_split(split: str, local_dir: str):
    parquet_prefix = SPLIT_MAP[split]
    out_dir = Path(local_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nListing parquet files for split '{split}' ...")
    all_files = list(list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=PARQUET_BRANCH,
    ))
    parquet_files = sorted([f for f in all_files if f.startswith(parquet_prefix) and f.endswith(".parquet")])

    if not parquet_files:
        print(f"  No parquet files found for split '{split}' under '{parquet_prefix}'!")
        return

    print(f"  Found {len(parquet_files)} parquet file(s) for '{split}':")
    for f in parquet_files:
        print(f"    {f}")

    for remote_path in parquet_files:
        filename = Path(remote_path).name
        dest = out_dir / filename
        if dest.exists():
            print(f"  [skip] {filename} already exists.")
            continue
        print(f"  Downloading {filename} -> {dest} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote_path,
            revision=PARQUET_BRANCH,
            local_dir=str(Path(local_dir)),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download saves to local_dir/remote_path; move to flat out_dir
        downloaded = Path(local_dir) / remote_path
        if downloaded.exists() and downloaded != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded.rename(dest)

    print(f"  Done! '{split}' parquet files saved to: {out_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RealBokeh_3MP parquet files from HuggingFace")
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "validation", "test", "all"],
        help="Which split to download (default: train)"
    )
    parser.add_argument(
        "--local_dir", type=str, default=LOCAL_DIR,
        help=f"Local directory to save parquet files (default: {LOCAL_DIR})"
    )
    args = parser.parse_args()

    splits = list(SPLIT_MAP.keys()) if args.split == "all" else [args.split]
    for split in splits:
        download_split(split, args.local_dir)

    print(f"\nAll done. Parquet files are in: {Path(args.local_dir).absolute()}")
    print("To use for training, run:")
    print(f"  .venv\\Scripts\\python.exe train.py -size small -data_path {args.local_dir} --parquet")
