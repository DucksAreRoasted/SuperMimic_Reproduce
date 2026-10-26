"""Differentiable forward kinematics read off a compiled MuJoCo model.

The joint convention is checked against ``mj_kinematics`` at construction, so G1, R1,
and K1 share one implementation.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from supermimic.mathutil import axis_angle_rot, quat_to_rot


class KinTree:
    def __init__(self, model: mujoco.MjModel, device: torch.device | str = "cpu"):
        requested = torch.device(device)
        self._requested = requested
        self.device = torch.device("cpu")
        self.nbody = int(model.nbody)
        self.nq = int(model.nq)
        self.nv = int(model.nv)
        self.n_dof = self.nq - 7
        parent = torch.tensor(model.body_parentid, dtype=torch.long, device=self.device)
        body_pos = torch.tensor(model.body_pos, dtype=torch.float32, device=self.device)
        body_quat = torch.tensor(model.body_quat, dtype=torch.float32, device=self.device)
        self.parent = parent
        self.body_pos = body_pos
        self.body_rot = quat_to_rot(body_quat)
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.nbody)]
        self.name_to_id = {n: i for i, n in enumerate(self.names)}
        joints = []
        for j in range(model.njnt):
            joints.append(
                {
                    "body": int(model.jnt_bodyid[j]),
                    "type": int(model.jnt_type[j]),
                    "qadr": int(model.jnt_qposadr[j]),
                    "dadr": int(model.jnt_dofadr[j]),
                    "axis": torch.tensor(model.jnt_axis[j], dtype=torch.float32, device=self.device),
                    "anchor": torch.tensor(model.jnt_pos[j], dtype=torch.float32, device=self.device),
                }
            )
        self.joints = joints
        self.body_joints: list[list[int]] = [[] for _ in range(self.nbody)]
        for j, joint in enumerate(joints):
            self.body_joints[joint["body"]].append(j)
        self.free_type = int(mujoco.mjtJoint.mjJNT_FREE)
        self.hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        self.convention = 0
        self._calibrate(model)

    def _numpy_pose(self, qpos: np.ndarray, convention: int) -> tuple[np.ndarray, np.ndarray]:
        q = torch.tensor(qpos, dtype=torch.float32).unsqueeze(0)
        xpos, xmat, _, _ = self._forward(q, None, convention)
        return xpos[0].numpy(), xmat[0].numpy()

    def _calibrate(self, model: mujoco.MjModel) -> int:
        data = mujoco.MjData(model)
        samples = [model.qpos0.copy()]
        q = model.qpos0.copy()
        # Move the first hinge if there is one.
        for joint in self.joints:
            if joint["type"] == self.hinge_type:
                q = q.copy()
                q[joint["qadr"]] += 0.4
                samples.append(q)
                break
        best = None
        for conv in (0, 1, 2):
            ok = True
            for qpos in samples:
                data.qpos[:] = qpos
                mujoco.mj_kinematics(model, data)
                try:
                    xpos, xmat = self._numpy_pose(qpos, conv)
                except RuntimeError:
                    ok = False
                    break
                if not (np.allclose(xpos, data.xpos, atol=1e-4) and np.allclose(xmat, data.xmat.reshape(-1, 3, 3), atol=1e-4)):
                    ok = False
                    break
            if ok:
                best = conv
                break
        if best is None:
            data.qpos[:] = model.qpos0
            mujoco.mj_kinematics(model, data)
            errs = []
            for conv in (0, 1, 2):
                xpos, _ = self._numpy_pose(model.qpos0, conv)
                errs.append(float(np.abs(xpos - data.xpos).max()))
            raise RuntimeError(f"FK convention mismatch, max errors {errs}")
        self.convention = best
        if self._requested.type != "cpu":
            self.to(self._requested)
        return best

    def _forward(self, qpos: torch.Tensor, qvel: torch.Tensor | None, convention: int):
        b = qpos.shape[0]
        pos_list = [qpos.new_zeros(b, 3)]
        rot_list = [torch.eye(3, device=qpos.device, dtype=qpos.dtype).expand(b, 3, 3)]
        lin_list = [qpos.new_zeros(b, 3)]
        ang_list = [qpos.new_zeros(b, 3)]
        for i in range(1, self.nbody):
            pid = int(self.parent[i])
            rot = rot_list[pid] @ self.body_rot[i]
            pos = pos_list[pid] + torch.matmul(rot_list[pid], self.body_pos[i])
            w = ang_list[pid]
            v = lin_list[pid] + torch.cross(ang_list[pid], pos - pos_list[pid], dim=-1)
            for jid in self.body_joints[i]:
                joint = self.joints[jid]
                if joint["type"] == self.free_type:
                    # MuJoCo writes the free-joint pose in the parent frame. qpos0 already
                    # contains the body offset, so the joint replaces that offset.
                    pos = qpos[:, joint["qadr"] : joint["qadr"] + 3]
                    rot = quat_to_rot(qpos[:, joint["qadr"] + 3 : joint["qadr"] + 7])
                    if qvel is not None:
                        v = qvel[:, joint["dadr"] : joint["dadr"] + 3]
                        w_body = qvel[:, joint["dadr"] + 3 : joint["dadr"] + 6]
                        w = torch.matmul(rot, w_body.unsqueeze(-1)).squeeze(-1)
                elif joint["type"] == self.hinge_type:
                    angle = qpos[:, joint["qadr"]]
                    axis = joint["axis"]
                    local = axis_angle_rot(axis, angle)
                    if convention == 0:
                        rot = rot @ local
                    else:
                        rot = local @ rot
                    if qvel is not None:
                        axis_w = torch.matmul(rot, axis)
                        qd = qvel[:, joint["dadr"]]
                        w = w + axis_w * qd.unsqueeze(-1)
            pos_list.append(pos)
            rot_list.append(rot)
            lin_list.append(v)
            ang_list.append(w)
        return (
            torch.stack(pos_list, dim=1),
            torch.stack(rot_list, dim=1),
            torch.stack(lin_list, dim=1),
            torch.stack(ang_list, dim=1),
        )

    def forward(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None):
        if qpos.ndim == 1:
            xpos, xmat, lin, ang = self._forward(qpos.unsqueeze(0), None if qvel is None else qvel.unsqueeze(0), self.convention)
            return xpos[0], xmat[0], None if qvel is None else lin[0], None if qvel is None else ang[0]
        return self._forward(qpos, qvel, self.convention)

    def body_ids(self, names: tuple[str, ...] | list[str]) -> torch.Tensor:
        return torch.tensor([self.name_to_id[n] for n in names], dtype=torch.long, device=self.device)

    def to(self, device: torch.device | str) -> "KinTree":
        self.device = torch.device(device)
        self.parent = self.parent.to(self.device)
        self.body_pos = self.body_pos.to(self.device)
        self.body_rot = self.body_rot.to(self.device)
        for joint in self.joints:
            joint["axis"] = joint["axis"].to(self.device)
            joint["anchor"] = joint["anchor"].to(self.device)
        return self
