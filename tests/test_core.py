"""Numerical checks for FK, the analytic reward path, and one training update."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supermimic.config import TrainConfig, imag_chunk_size
from supermimic.data.filter_bones import feasibility_flags, name_rejected
from supermimic.data.motion import MotionLibrary, sine_clip, standing_clip
from supermimic.data.retarget import pairs_for, retarget_clip
from supermimic.deploy.export import export_policy
from supermimic.deploy.runtime import TrackerRuntime
from supermimic.kinematics.fk import KinTree
from supermimic.mdp.observations import integrate_delta
from supermimic.robots.actuators import G1_5020, g1_actuator
from supermimic.robots.registry import get_robot
from supermimic.training.loop import Trainer


class TestRobots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robots = {name: get_robot(name) for name in ("g1", "r1", "k1")}

    def test_degrees_of_freedom(self):
        self.assertEqual(self.robots["g1"][0].n_dof, 29)
        self.assertEqual(self.robots["r1"][0].n_dof, 24)
        self.assertEqual(self.robots["k1"][0].n_dof, 22)
        self.assertEqual(self.robots["g1"][0].obs_dim, 93)
        self.assertEqual(self.robots["g1"][0].goal_frame_dim, 68)
        self.assertEqual(self.robots["g1"][0].xi_dim, 33)

    def test_g1_action_scale_matches_formula(self):
        knee = g1_actuator("left_knee_joint")
        self.assertAlmostEqual(knee.action_scale, 0.25 * knee.effort / knee.kp, places=6)
        self.assertAlmostEqual(G1_5020.kp, G1_5020.armature * (2 * 3.1415926535 * 10) ** 2, places=3)

    def test_fk_matches_mujoco(self):
        for name, (profile, model) in self.robots.items():
            kin = KinTree(model)
            rng = np.random.default_rng(0)
            qpos = model.qpos0.copy()
            qpos[7:] += rng.uniform(-0.2, 0.2, size=profile.n_dof)
            qvel = rng.normal(scale=0.2, size=model.nv)
            data = mujoco.MjData(model)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_kinematics(model, data)
            xpos, xmat, _, _ = kin.forward(torch.tensor(qpos, dtype=torch.float32))
            self.assertTrue(np.allclose(xpos.numpy(), data.xpos, atol=1e-4), name)
            self.assertTrue(np.allclose(xmat.numpy(), data.xmat.reshape(-1, 3, 3), atol=1e-4), name)

    def test_zero_residual_is_identity(self):
        profile, _ = self.robots["g1"]
        state = torch.tensor(np.concatenate([profile.qpos0, np.zeros(6 + profile.n_dof)]), dtype=torch.float32)
        delta = torch.zeros(12 + 2 * profile.n_dof)
        out = integrate_delta(state, delta, profile.n_dof)
        self.assertTrue(torch.allclose(out, state, atol=1e-5))

    def test_stand_for_a_second(self):
        profile, model = self.robots["g1"]
        data = mujoco.MjData(model)
        data.qpos[:] = profile.qpos0
        mujoco.mj_forward(model, data)
        z0 = float(data.qpos[2])
        for _ in range(50):
            data.ctrl[:] = profile.qpos0[7:]
            for _s in range(4):
                mujoco.mj_step(model, data)
        self.assertGreater(float(data.qpos[2]), z0 * 0.5)

    def test_sdk_roundtrip_shape(self):
        for profile, _ in self.robots.values():
            self.assertEqual(len(profile.sdk_indices), profile.n_dof)
            self.assertLess(max(profile.sdk_indices), profile.sdk_width)


class TestData(unittest.TestCase):
    def test_name_filter_keeps_flips(self):
        self.assertTrue(name_rejected("subject_stair_walk.csv"))
        self.assertFalse(name_rejected("flip_360_inplace.csv"))
        self.assertFalse(name_rejected("side_roll_01.csv"))
        self.assertEqual(feasibility_flags(-0.2, 0.0, 0.0, 0.0, False, False, False), ["ground_penetration"])

    def test_retarget_r1_and_k1(self):
        g1, g1_model = get_robot("g1")
        clip = standing_clip(g1, seconds=0.2)
        clip.qpos = clip.qpos[:5]
        clip.qvel = clip.qvel[:5]
        for name in ("r1", "k1"):
            target, model = get_robot(name)
            pairs = pairs_for(g1, target)
            out = retarget_clip(g1_model, g1, model, target, clip, pairs, iterations=3)
            self.assertEqual(out.qpos.shape[0], 5)
            self.assertTrue(np.isfinite(out.qpos).all())


class TestTrain(unittest.TestCase):
    def test_one_update_and_export(self):
        profile, model = get_robot("g1")
        kin = KinTree(model)
        library = MotionLibrary([standing_clip(profile, 2.0), sine_clip(profile, 2.0, amp=0.05)])
        cfg = TrainConfig(
            num_envs=2,
            buffer_steps=32,
            k=2,
            horizon=2,
            wm_batches=1,
            wm_batch=2,
            imag_batch=2,
            total_env_steps=6,
            hidden_dyn=(32, 32),
            hidden_enc=(32, 32),
            hidden_dec=(32, 32),
            hidden_critic=(32,),
            hidden_term=(32,),
            device="cpu",
            memory_budget=256 * 1024**2,
            train_velocity=True,
            log_dir="logs/test",
            future_frames=2,
            history=4,
        )
        trainer = Trainer(cfg, profile, str(profile.scene_xml), library, kin)
        trainer.env.pose_noise = 0.0
        logs = trainer.train()
        self.assertTrue(logs)
        self.assertTrue(np.isfinite(logs[-1]["policy_loss"]))
        path = export_policy(trainer.policy, Path("logs/test/policy"), profile.obs_dim, cfg.future_frames * profile.goal_frame_dim, cfg.history)
        fresh = trainer.policy
        state = {k: v.detach().clone() for k, v in fresh.state_dict().items()}
        fresh.load_state_dict(torch.load(path))
        for key, value in state.items():
            self.assertTrue(torch.allclose(value, fresh.state_dict()[key]))
        runtime = TrackerRuntime(fresh, profile)
        q = np.zeros(profile.sdk_width)
        qd = np.zeros(profile.sdk_width)
        goal = np.zeros(cfg.future_frames * profile.goal_frame_dim)
        out = runtime.act(q, qd, np.zeros(3), np.array([1.0, 0, 0, 0]), goal)
        self.assertEqual(out["q_sdk"].shape, (profile.sdk_width,))

    def test_chunk_math(self):
        self.assertGreater(imag_chunk_size(40 * 1024**3, 10 * 1024**3, 4096), 1000)
        self.assertEqual(imag_chunk_size(2 * 1024**3, 2 * 1024**3 - 1000, 4096), 1)


if __name__ == "__main__":
    unittest.main()
