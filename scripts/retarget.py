"""Retarget G1 motions onto Unitree R1 and Booster K1.

One file:
    python scripts/retarget.py --csv data/lafan1/g1/walk1_subject1.csv --target r1 --out data/retarget/r1/walk.npz

A directory (csv or npz), both robots, skipping files that already exist:
    python scripts/retarget.py --input data/lafan1/g1 --targets r1,k1 --out data/retarget/lafan1 --jobs 8

Outputs are 50 Hz npz with qpos (wxyz), qvel, and name. scripts/train.py reads that directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _collect(folder: Path) -> list[Path]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".csv", ".npz"} and p.is_file())
    if not files:
        raise SystemExit(f"no csv or npz files in {folder}")
    return files


def _retarget_one(src: str, target_name: str, out: str, src_fps: float, iterations: int) -> str:
    from supermimic.data.motion import load_motion_file
    from supermimic.data.retarget import pairs_for, retarget_clip
    from supermimic.robots.registry import get_robot

    source, source_model = get_robot("g1")
    target, target_model = get_robot(target_name)
    clip = load_motion_file(Path(src), source.n_dof, src_fps=src_fps)
    retargeted = retarget_clip(
        source_model,
        source,
        target_model,
        target,
        clip,
        pairs_for(source, target),
        iterations=iterations,
    )
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, qpos=retargeted.qpos, qvel=retargeted.qvel, name=retargeted.name)
    return f"{path} frames={retargeted.length}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="one G1 csv")
    parser.add_argument("--input", type=Path, help="directory of G1 csv or npz clips")
    parser.add_argument("--target", choices=("r1", "k1"), help="used with --csv")
    parser.add_argument("--targets", default="r1,k1", help="comma-separated robots, used with --input")
    parser.add_argument("--out", type=Path, required=True, help="one npz, or the directory that will contain <robot>/")
    parser.add_argument("--src-fps", type=float, default=30.0, help="csv frame rate before resampling to 50 Hz")
    parser.add_argument("--iterations", type=int, default=8, help="IK iterations per frame")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="retarget at most this many source clips; 0 means all")
    parser.add_argument("--force", action="store_true", help="overwrite existing npz files")
    args = parser.parse_args()

    if args.csv is not None:
        if args.target is None:
            raise SystemExit("--csv requires --target")
        print(_retarget_one(str(args.csv), args.target, str(args.out), args.src_fps, args.iterations))
        return
    if args.input is None:
        raise SystemExit("pass --csv or --input")

    targets = [part.strip() for part in args.targets.split(",") if part.strip()]
    for name in targets:
        if name not in ("r1", "k1"):
            raise SystemExit(f"unknown target {name}")
    files = _collect(args.input)
    if args.limit:
        files = files[: args.limit]
    tasks = []
    for src in files:
        for target in targets:
            dest = args.out / target / f"{src.stem}.npz"
            if dest.exists() and not args.force:
                continue
            tasks.append((str(src), target, str(dest), args.src_fps, args.iterations))
    print(f"{len(files)} source clips, {len(tasks)} retarget jobs, {args.jobs} workers", flush=True)
    if not tasks:
        print("nothing to do")
        return
    if args.jobs == 1:
        for i, task in enumerate(tasks, start=1):
            print(f"[{i}/{len(tasks)}] {_retarget_one(*task)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_retarget_one, *task) for task in tasks]
        for i, future in enumerate(as_completed(futures), start=1):
            print(f"[{i}/{len(tasks)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
