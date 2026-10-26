"""Save the tracker. ONNX is attempted; the state dict is the source of truth for tests."""

from __future__ import annotations

from pathlib import Path

import torch

from supermimic.policy.networks import Policy


def export_policy(policy: Policy, path: Path, obs_dim: int, goal_dim: int, history: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), path.with_suffix(".pt"))
    wrapper = _Wrapper(policy).eval()
    device = next(policy.parameters()).device
    hist = torch.zeros(1, history, obs_dim, device=device)
    goal = torch.zeros(1, goal_dim, device=device)
    try:
        torch.onnx.export(wrapper, (hist, goal), str(path.with_suffix(".onnx")), input_names=["history", "goal"], output_names=["action"], dynamo=False)
    except TypeError:
        try:
            torch.onnx.export(wrapper, (hist, goal), str(path.with_suffix(".onnx")), input_names=["history", "goal"], output_names=["action"])
        except Exception as exc:  # pragma: no cover - version dependent
            path.with_suffix(".onnx.txt").write_text(f"onnx export skipped: {exc}\n")
    except Exception as exc:  # pragma: no cover
        path.with_suffix(".onnx.txt").write_text(f"onnx export skipped: {exc}\n")
    return path.with_suffix(".pt")


class _Wrapper(torch.nn.Module):
    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def forward(self, hist, goal):
        action, _ = self.policy.act(hist, goal, "motion")
        return action
