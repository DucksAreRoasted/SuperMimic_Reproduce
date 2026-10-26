"""Download the motion sets from ModelScope.

LAFAN1 (40 G1 clips):
    unitreerobotics/LAFAN1_Retargeting_Dataset  ->  data/lafan1/g1/*.csv
    https://www.modelscope.cn/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset

BONES-SEED (G1 retarget, g1.tar.gz is about 23 GB):
    bones-studio/seed  ->  data/bones/g1.tar.gz
    https://www.modelscope.cn/datasets/bones-studio/seed

Both are public on ModelScope. A token is optional and only used when the
environment variable MODELSCOPE_TOKEN is set (higher rate limits).

Examples:
    python scripts/download_datasets.py --dataset lafan1
    python scripts/download_datasets.py --dataset bones
    python scripts/download_datasets.py --dataset all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LAFAN_REPO = "unitreerobotics/LAFAN1_Retargeting_Dataset"
BONES_REPO = "bones-studio/seed"


def _snapshot():
    try:
        from modelscope_hub.compat.snapshot_download import dataset_snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "modelscope-hub is not installed. Run: pip install -e '.[data]'   or   pip install modelscope-hub"
        ) from exc
    return dataset_snapshot_download


def _token() -> str | None:
    return os.environ.get("MODELSCOPE_TOKEN") or os.environ.get("MODELSCOPE_API_TOKEN") or None


def download_lafan1(dest: Path, repo: str) -> None:
    snapshot = _snapshot()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"ModelScope dataset {repo} -> {dest}", flush=True)
    snapshot(
        repo,
        local_dir=str(dest),
        allow_patterns=["g1/*.csv"],
        token=_token(),
    )
    n = len(list((dest / "g1").glob("*.csv")))
    print(f"LAFAN1: {n} csv files in {dest / 'g1'}")
    if n != 40:
        print(f"warning: the paper uses 40 LAFAN1 clips, found {n}", file=sys.stderr)


def download_bones(dest: Path, repo: str) -> None:
    snapshot = _snapshot()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"ModelScope dataset {repo} -> {dest}", flush=True)
    print("g1.tar.gz is about 23 GB; the download resumes if it is interrupted.", flush=True)
    snapshot(
        repo,
        local_dir=str(dest),
        allow_patterns=["g1.tar.gz", "metadata/seed_metadata_v004.csv"],
        token=_token(),
    )
    archive = dest / "g1.tar.gz"
    if not archive.exists():
        raise SystemExit(f"download finished but {archive} is missing")
    print(f"BONES-SEED: {archive} ({archive.stat().st_size / 1e9:.1f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("lafan1", "bones", "all"), default="all")
    parser.add_argument("--dest", type=Path, default=ROOT / "data", help="root data directory")
    parser.add_argument("--lafan-repo", default=LAFAN_REPO)
    parser.add_argument("--bones-repo", default=BONES_REPO)
    args = parser.parse_args()
    if args.dataset in ("lafan1", "all"):
        download_lafan1(args.dest / "lafan1", args.lafan_repo)
    if args.dataset in ("bones", "all"):
        download_bones(args.dest / "bones", args.bones_repo)


if __name__ == "__main__":
    main()
