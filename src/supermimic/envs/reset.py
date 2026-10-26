"""Return-based reset sampling. Bins are one second of motion."""

from __future__ import annotations

import numpy as np

from supermimic.data.motion import MotionLibrary


class ResetSampler:
    def __init__(self, library: MotionLibrary, gmax: float = 10.0, alpha: float = 0.05, hist_alpha: float = 0.01):
        self.library = library
        self.gmax = gmax
        self.alpha = alpha
        self.hist_alpha = hist_alpha
        self.G = np.full(library.n_bins, gmax, dtype=np.float64)
        self.hist = np.full(library.n_bins, 1.0 / library.n_bins, dtype=np.float64)

    def update(self, bins: list[int], returns: list[float]) -> None:
        for b, r in zip(bins, returns):
            self.G[b] = (1 - self.alpha) * self.G[b] + self.alpha * r
        logits = -25.0 * self.G / self.gmax
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        self.hist = (1 - self.hist_alpha) * self.hist + self.hist_alpha * p

    def sample(self, rng: np.random.Generator) -> tuple[int, int]:
        u = rng.random()
        if u < 0.5:
            p = np.full(self.library.n_bins, 1.0 / self.library.n_bins)
        elif u < 0.9:
            logits = -25.0 * self.G / self.gmax
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
        else:
            p = self.hist / self.hist.sum()
        bin_id = int(rng.choice(self.library.n_bins, p=p))
        clip = 0
        local = bin_id
        for i, offset in enumerate(self.library.bin_offsets):
            nxt = self.library.bin_offsets[i + 1] if i + 1 < len(self.library.bin_offsets) else self.library.n_bins
            if offset <= bin_id < nxt:
                clip = i
                local = bin_id - offset
                break
        motion = self.library.clips[clip]
        start = local * self.library.fps
        stop = min(motion.length - 1, start + self.library.fps)
        frame = int(rng.integers(start, max(start + 1, stop)))
        return clip, frame
