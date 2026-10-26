"""CPU MuJoCo vector environment. One compiled model copy per world so friction and COM differ."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from supermimic.robots.profile import RobotProfile


class VectorSim:
    def __init__(self, model: mujoco.MjModel, profile: RobotProfile, num_envs: int, scene_xml: Path):
        self.profile = profile
        self.num_envs = num_envs
        self.nq = model.nq
        self.nv = model.nv
        self.nu = profile.n_dof
        self.substeps = 4
        self.models: list[mujoco.MjModel] = []
        self.datas: list[mujoco.MjData] = []
        self.foot_ids: list[list[int]] = []
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, profile.torso_body)
        xml = Path(scene_xml).read_text()
        del model  # every world gets its own compile so domain randomization cannot alias
        for _ in range(num_envs):
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            self.models.append(m)
            self.datas.append(d)
            ids = []
            for name in ("left_foot_collision", "right_foot_collision"):
                ids.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name))
            self.foot_ids.append(ids)

    def set_state(self, env: int, qpos: np.ndarray, qvel: np.ndarray) -> None:
        data = self.datas[env]
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(self.models[env], data)

    def get_state(self, env: int) -> np.ndarray:
        data = self.datas[env]
        return np.concatenate([data.qpos, data.qvel]).astype(np.float32)

    def states(self) -> np.ndarray:
        return np.stack([self.get_state(i) for i in range(self.num_envs)], axis=0)

    def set_friction(self, env: int, mu: float) -> None:
        model = self.models[env]
        for gid in self.foot_ids[env]:
            model.geom_friction[gid, 0] = mu

    def set_com_offset(self, env: int, offset: np.ndarray) -> None:
        self.models[env].body_ipos[self.torso_id] = self.models[env].body_ipos[self.torso_id] + offset
        mujoco.mj_setConst(self.models[env], self.datas[env])

    def step(self, targets: np.ndarray) -> None:
        for i, data in enumerate(self.datas):
            data.ctrl[:] = targets[i]
            for _ in range(self.substeps):
                mujoco.mj_step(self.models[i], data)
