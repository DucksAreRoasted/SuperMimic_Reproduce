"""Score a saved policy on a motion directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from supermimic.config import TrainConfig
from supermimic.data.motion import load_directory
from supermimic.kinematics.fk import KinTree
from supermimic.policy.networks import Policy
from supermimic.robots.registry import get_robot
from supermimic.training.evaluate import evaluate_library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="g1", choices=["g1", "r1", "k1"])
    parser.add_argument("--motions", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    profile, model = get_robot(args.robot)
    kin = KinTree(model, device=args.device)
    cfg = TrainConfig()
    if args.hidden != 2048:
        cfg.hidden_enc = (args.hidden, args.hidden)
        cfg.hidden_dec = (args.hidden, args.hidden)
    goal_dim = cfg.future_frames * profile.goal_frame_dim
    policy = Policy(profile.obs_dim, goal_dim, profile.n_dof, cfg.history, cfg.latent_dim, cfg.hidden_enc, cfg.hidden_dec)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    policy.to(args.device)
    library = load_directory(Path(args.motions), profile.n_dof)
    print(evaluate_library(policy, profile, kin, library, torch.device(args.device), cfg.future_frames))


if __name__ == "__main__":
    main()
