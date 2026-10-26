"""Train a SuperMimic tracker. Defaults are the paper's one-GPU batch on an L40 budget."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from supermimic.config import TrainConfig
from supermimic.data.motion import MotionLibrary, load_directory, load_npz_directory, sine_clip, standing_clip
from supermimic.kinematics.fk import KinTree
from supermimic.robots.registry import get_robot
from supermimic.training.evaluate import evaluate_library
from supermimic.training.loop import Trainer


def build_config(args) -> TrainConfig:
    hidden = (args.hidden, args.hidden) if args.hidden else None
    cfg = TrainConfig(
        num_envs=args.num_envs,
        imag_batch=args.imag_batch,
        wm_batch=args.wm_batch,
        wm_batches=args.wm_batches,
        total_env_steps=args.steps,
        buffer_steps=args.buffer,
        horizon=args.horizon,
        k=args.k,
        device=args.device,
        log_dir=args.log_dir,
        seed=args.seed,
        train_velocity=args.velocity,
    )
    if hidden is not None:
        cfg.hidden_dyn = (args.hidden, args.hidden)
        cfg.hidden_enc = (args.hidden, args.hidden)
        cfg.hidden_dec = (args.hidden, args.hidden)
        cfg.hidden_critic = (args.hidden,)
        cfg.hidden_term = (args.hidden,)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="g1", choices=["g1", "r1", "k1"])
    parser.add_argument("--motions", default="synthetic", help="synthetic, or a directory of LAFAN csvs")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--imag-batch", type=int, default=4096)
    parser.add_argument("--wm-batch", type=int, default=2048)
    parser.add_argument("--wm-batches", type=int, default=10)
    parser.add_argument("--steps", type=int, default=320_000)
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=0, help="override network width; 0 keeps the paper widths")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", default="logs/run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--velocity", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    profile, model = get_robot(args.robot)
    kin = KinTree(model)
    if args.motions == "synthetic":
        library = MotionLibrary([standing_clip(profile, 4.0), sine_clip(profile, 4.0)])
    else:
        motion_path = Path(args.motions)
        if list(motion_path.glob("*.csv")):
            library = load_directory(motion_path, profile.n_dof)
        else:
            library = load_npz_directory(motion_path, profile.n_dof)
    cfg = build_config(args)
    trainer = Trainer(cfg, profile, str(profile.scene_xml), library, kin)
    trainer.env.pose_noise = 0.05
    logs = trainer.train()
    if logs:
        for row in logs:
            if int(row["step"]) % 10 == 0 or int(row["step"]) == logs[-1]["step"]:
                print(row, flush=True)
    else:
        print("no updates", flush=True)
    from supermimic.deploy.export import export_policy

    export_policy(
        trainer.policy,
        Path(args.log_dir) / "policy",
        profile.obs_dim,
        cfg.future_frames * profile.goal_frame_dim,
        cfg.history,
    )
    if args.eval:
        metrics = evaluate_library(trainer.policy, profile, trainer.kin, library, trainer.device, cfg.future_frames)
        print(metrics)


if __name__ == "__main__":
    main()
