"""Batched quaternion and rotation helpers. Quaternions are wxyz."""

from __future__ import annotations

import torch

def _unsq(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    while v.ndim < ref.ndim:
        v = v.unsqueeze(0)
    return v


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz
    row0 = torch.stack((r00, r01, r02), dim=-1)
    row1 = torch.stack((r10, r11, r12), dim=-1)
    row2 = torch.stack((r20, r21, r22), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def rot_to_quat(r: torch.Tensor) -> torch.Tensor:
    """Shepperd's method. r is [..., 3, 3]."""
    batch = r.shape[:-2]
    m00, m01, m02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    m10, m11, m12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    m20, m21, m22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]
    trace = m00 + m11 + m22
    q = r.new_zeros(batch + (4,))
    c0 = trace > 0
    c1 = (m00 >= m11) & (m00 >= m22) & ~c0
    c2 = (m11 > m00) & (m11 >= m22) & ~c0
    c3 = ~(c0 | c1 | c2)
    s = torch.sqrt((trace + 1).clamp_min(0)).unsqueeze(-1) * 2
    q = torch.where(
        c0.unsqueeze(-1),
        torch.stack((s.squeeze(-1) / 4, (m21 - m12) / s.squeeze(-1), (m02 - m20) / s.squeeze(-1), (m10 - m01) / s.squeeze(-1)), dim=-1),
        q,
    )
    s1 = torch.sqrt((1 + m00 - m11 - m22).clamp_min(0)) * 2
    q = torch.where(
        c1.unsqueeze(-1),
        torch.stack(((m21 - m12) / s1, s1 / 4, (m01 + m10) / s1, (m02 + m20) / s1), dim=-1),
        q,
    )
    s2 = torch.sqrt((1 + m11 - m00 - m22).clamp_min(0)) * 2
    q = torch.where(
        c2.unsqueeze(-1),
        torch.stack(((m02 - m20) / s2, (m01 + m10) / s2, s2 / 4, (m12 + m21) / s2), dim=-1),
        q,
    )
    s3 = torch.sqrt((1 + m22 - m00 - m11).clamp_min(0)) * 2
    q = torch.where(
        c3.unsqueeze(-1),
        torch.stack(((m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, s3 / 4), dim=-1),
        q,
    )
    return quat_normalize(q)


def yaw_of_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotz(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    row0 = torch.stack((c, -s, z), dim=-1)
    row1 = torch.stack((s, c, z), dim=-1)
    row2 = torch.stack((z, z, o), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def rot6d_of(r: torch.Tensor) -> torch.Tensor:
    return r[..., :, :2].reshape(r.shape[:-2] + (6,))


def rotvec_to_quat(rv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = rv.norm(dim=-1, keepdim=True).clamp_min(eps)
    half = 0.5 * theta
    axis = rv / theta
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    q = torch.cat((w, xyz), dim=-1)
    small = (rv.norm(dim=-1) < 1e-6).unsqueeze(-1)
    ident = torch.zeros_like(q)
    ident[..., 0] = 1
    return torch.where(small, ident, q)


def rot_log(r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """vee(log R), shape [..., 3]."""
    tr = (r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]).clamp(-1 + 1e-6, 3)
    cos = ((tr - 1) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos)
    w = torch.stack(
        (
            r[..., 2, 1] - r[..., 1, 2],
            r[..., 0, 2] - r[..., 2, 0],
            r[..., 1, 0] - r[..., 0, 1],
        ),
        dim=-1,
    )
    denom = (2 * torch.sin(theta)).unsqueeze(-1).clamp_min(eps)
    rv = theta.unsqueeze(-1) * w / denom
    small = (theta < 1e-4).unsqueeze(-1)
    return torch.where(small, 0.5 * w, rv)


def axis_angle_rot(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """axis [3] or [..., 3], angle [...]. Right-handed."""
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if angle.ndim == axis.ndim - 1:
        angle = angle.unsqueeze(-1)
    if axis.ndim < angle.ndim:
        axis = _unsq(axis, angle)
    x, y, z = axis.unbind(-1)
    c = torch.cos(angle.squeeze(-1))
    s = torch.sin(angle.squeeze(-1))
    oc = 1 - c
    r00 = c + x * x * oc
    r01 = x * y * oc - z * s
    r02 = x * z * oc + y * s
    r10 = y * x * oc + z * s
    r11 = c + y * y * oc
    r12 = y * z * oc - x * s
    r20 = z * x * oc - y * s
    r21 = z * y * oc + x * s
    r22 = c + z * z * oc
    return torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )


def projected_gravity(rot: torch.Tensor) -> torch.Tensor:
    g = rot.new_tensor([0.0, 0.0, -1.0])
    return torch.matmul(rot.transpose(-1, -2), g)


def yaw_from_rot(rot: torch.Tensor) -> torch.Tensor:
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
