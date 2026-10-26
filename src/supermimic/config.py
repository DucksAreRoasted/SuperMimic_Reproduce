"""Paper hyperparameters (Table A.5) and the single-L40 memory budget."""

from __future__ import annotations

from dataclasses import dataclass, field


L40_BUDGET_BYTES = 40 * 1024**3


@dataclass
class TrainConfig:
    """Defaults match the paper on one GPU. Imagination is chunked to fit the budget."""

    num_envs: int = 4096
    buffer_steps: int = 1000
    k: int = 8
    horizon: int = 32
    wm_batches: int = 10
    wm_batch: int = 2048
    imag_batch: int = 4096
    lr: float = 2e-4
    policy_betas: tuple[float, float] = (0.9, 0.95)
    total_env_steps: int = 320_000
    updates_per_env_step: float = 0.4
    cosine_fraction: float = 250_000 / 320_000
    lr_final_scale: float = 0.03
    gamma: float = 0.99
    lam: float = 0.95
    action_noise_std: float = 0.1
    history: int = 10
    future_frames: int = 10
    control_dt: float = 0.02
    sim_dt: float = 0.005
    hidden_dyn: tuple[int, ...] = (2048, 2048, 2048)
    hidden_enc: tuple[int, ...] = (2048, 1024, 512, 256)
    hidden_dec: tuple[int, ...] = (2048, 1024, 512, 256)
    hidden_critic: tuple[int, ...] = (256, 128)
    hidden_term: tuple[int, ...] = (256, 256)
    latent_dim: int = 32
    critic_epochs: int = 5
    critic_minibatches: int = 4
    grad_clip: float = 1.0
    policy_grad_element_clip: float = 0.1
    term_loss_weight: float = 1.0
    seed: int = 0
    device: str = "cuda"
    memory_budget: int | None = None
    train_velocity: bool = False
    velocity_fraction: float = 0.5
    latent_align_weight: float = 0.1
    episode_timeout_s: float = 10.0
    log_dir: str = "logs/run"
    eval_every: int = 4000
    # Nominal scales for the state residual and the state loss.
    sigma_pos: float = 0.05
    sigma_rot: float = 0.1
    sigma_lin: float = 0.5
    sigma_ang: float = 1.0
    sigma_q: float = 0.1
    sigma_qd: float = 2.0
    gmax_alpha: float = 0.05
    hist_alpha: float = 0.01
    reset_uniform: float = 0.5
    reset_softmax: float = 0.4
    softmax_temp: float = 25.0


def memory_budget_bytes(device: str, total_vram: int | None, override: int | None) -> int:
    """40 GB on an L40-class card. Smaller devices use 75% of VRAM, still chunking the same update."""
    if override is not None:
        return override
    if device == "cpu" or not total_vram:
        return 2 * 1024**3
    return min(L40_BUDGET_BYTES, int(total_vram * 0.75))


def imag_chunk_size(budget: int, allocated: int, imag_batch: int, bytes_per_rollout: int = 8 * 1024**2) -> int:
    remain = budget - allocated - (1 << 30)
    if remain <= 0:
        return 1
    return max(1, min(imag_batch, remain // max(bytes_per_rollout, 1)))
