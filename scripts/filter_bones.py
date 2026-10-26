"""Filter BONES-SEED the way Appendix E describes.

Two passes:
  1. Drop clip names that mention terrain or objects the robot cannot use.
     Flips and side rolls are kept.
  2. Optional MuJoCo feasibility check on extracted G1 clips (elevated support,
     ground penetration, stationary floating feet).

The motion archive is data/bones/g1.tar.gz from scripts/download_datasets.py.
Name filtering also works on a plain text list, one filename per line.

Examples:
    python scripts/filter_bones.py --names clips.txt --out kept.txt
    python scripts/filter_bones.py \\
        --archive data/bones/g1.tar.gz \\
        --out data/bones/kept.txt \\
        --extract data/bones/g1_filtered \\
        --feasibility
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supermimic.data.filter_bones import feasibility_flags, name_rejected


def _members(archive: Path):
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if member.isfile():
                yield member.name


def _extract(archive: Path, names: set[str], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if Path(member.name).name not in names and member.name not in names:
                continue
            member.name = Path(member.name).name
            if sys.version_info >= (3, 12):
                tar.extract(member, dest, filter="data")
            else:
                tar.extract(member, dest)
            written += 1
            if written % 1000 == 0:
                print(f"extracted {written}", flush=True)
    return written


def _longest_run(mask: np.ndarray, dt: float) -> float:
    best = current = 0.0
    for flag in mask:
        if flag:
            current += dt
            best = max(best, current)
        else:
            current = 0.0
    return best


def feasibility_of_qpos(model, qpos: np.ndarray, fps: float = 50.0) -> list[str]:
    """Appendix E on one 50 Hz G1 trajectory. qpos is MuJoCo order, wxyz quaternion."""
    import mujoco

    data = mujoco.MjData(model)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("left_foot_collision", "right_foot_collision")
    ]
    n = qpos.shape[0]
    zmin = np.empty(n)
    both_air = np.empty(n, dtype=bool)
    upright = np.empty(n, dtype=bool)
    standing = np.empty(n, dtype=bool)
    still = np.empty(n, dtype=bool)
    prev = None
    for i in range(n):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        zmin[i] = float(data.xpos[1:, 2].min())
        bottoms = []
        for gid in feet:
            half = float(model.geom_size[gid, 2] if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX else 0.0)
            bottoms.append(float(data.geom_xpos[gid, 2] - half))
        both_air[i] = all(z > 0.02 for z in bottoms)
        rot = data.xmat[pelvis].reshape(3, 3)
        grav_z = float(rot.T @ np.array([0.0, 0.0, -1.0]))[2]
        upright[i] = -grav_z > 0.9
        standing[i] = 0.60 <= float(data.xpos[pelvis, 2]) <= 0.85
        if prev is None:
            still[i] = False
        else:
            speed = float(np.linalg.norm(data.xpos[pelvis] - prev) * fps)
            qd = np.diff(qpos[i - 1 : i + 1, 7:], axis=0)[0] * fps
            still[i] = speed < 0.1 and float(np.sqrt(np.mean(qd**2))) < 0.5
        prev = data.xpos[pelvis].copy()
    dt = 1.0 / fps
    return feasibility_flags(
        body_z_min=float(zmin.min()),
        seconds_above_020=_longest_run(zmin > 0.20, dt),
        fraction_below_008=float(np.mean(zmin < -0.08)),
        both_feet_air_s=_longest_run(both_air & upright & standing & still, dt),
        pelvis_upright=True,
        pelvis_standing=True,
        nearly_still=True,
    )


def _qpos_of(path: Path, n_dof: int) -> np.ndarray:
    from supermimic.data.motion import load_motion_file

    return load_motion_file(path, n_dof).qpos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--names", type=Path, help="text file, one clip filename per line")
    parser.add_argument("--archive", type=Path, help="g1.tar.gz from the BONES-SEED download")
    parser.add_argument("--dir", type=Path, help="directory of already extracted clips")
    parser.add_argument("--out", type=Path, required=True, help="kept filenames, one per line")
    parser.add_argument("--extract", type=Path, help="write kept archive members here")
    parser.add_argument("--feasibility", action="store_true", help="run the Appendix E geometric checks")
    parser.add_argument("--rejected-out", type=Path, help="optional list of dropped names and reasons")
    args = parser.parse_args()
    if args.names is None and args.archive is None and args.dir is None:
        raise SystemExit("pass --names, --archive, or --dir")

    if args.names is not None:
        raw = [line.strip() for line in args.names.read_text().splitlines() if line.strip()]
    elif args.archive is not None:
        print(f"scanning {args.archive}", flush=True)
        raw = list(_members(args.archive))
        print(f"archive members: {len(raw)}", flush=True)
    else:
        raw = [str(p.relative_to(args.dir)) for p in sorted(args.dir.rglob("*")) if p.is_file()]

    kept = [name for name in raw if not name_rejected(Path(name).name)]
    dropped = [(name, "name") for name in raw if name_rejected(Path(name).name)]
    print(f"name filter: kept {len(kept)} dropped {len(dropped)} of {len(raw)}", flush=True)

    if args.extract is not None:
        if args.archive is None:
            raise SystemExit("--extract requires --archive")
        n = _extract(args.archive, {Path(name).name for name in kept}, args.extract)
        print(f"extracted {n} files to {args.extract}", flush=True)

    if args.feasibility:
        from supermimic.robots.registry import get_robot

        profile, model = get_robot("g1")
        folder = args.extract or args.dir
        if folder is None:
            raise SystemExit("--feasibility needs --extract or --dir")
        survived = []
        for i, name in enumerate(kept):
            path = folder / Path(name).name
            if not path.exists():
                dropped.append((name, "missing_after_extract"))
                continue
            if path.suffix.lower() not in {".csv", ".npz"}:
                survived.append(name)
                continue
            try:
                reasons = feasibility_of_qpos(model, _qpos_of(path, profile.n_dof))
            except Exception as exc:
                dropped.append((name, f"unreadable:{exc}"))
                continue
            if reasons:
                dropped.append((name, ",".join(reasons)))
            else:
                survived.append(name)
            if (i + 1) % 200 == 0:
                print(f"feasibility {i + 1}/{len(kept)} kept {len(survived)}", flush=True)
        kept = survived
        print(f"after feasibility: kept {len(kept)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(kept) + ("\n" if kept else ""))
    if args.rejected_out is not None:
        args.rejected_out.parent.mkdir(parents=True, exist_ok=True)
        args.rejected_out.write_text("\n".join(f"{name}\t{reason}" for name, reason in dropped) + ("\n" if dropped else ""))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
