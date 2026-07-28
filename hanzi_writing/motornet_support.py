"""Small MotorNet coordinate helpers shared by Hanzi environments and audits."""

from __future__ import annotations

import math

import numpy as np


def baseline_joint_state(effector) -> np.ndarray:
    position = np.asarray(effector.pos_range_bound, dtype=np.float64) * 0.5
    position += np.asarray(effector.pos_upper_bound, dtype=np.float64)
    position += np.asarray((0.1, 0.5), dtype=np.float64)
    return np.concatenate((position, np.zeros(2, dtype=np.float64)))


def baseline_anchor(effector) -> np.ndarray:
    import torch

    joint_state = baseline_joint_state(effector)
    with torch.no_grad():
        cartesian = effector.skeleton.joint2cartesian(
            torch.as_tensor(joint_state[None, :], dtype=torch.float32)
        )
    return cartesian.detach().cpu().numpy()[0, :2].astype(np.float64)


def inverse_kinematics(points: np.ndarray, effector) -> tuple[np.ndarray, np.ndarray]:
    """Solve the positive-elbow branch used by the project-2 baseline arm."""

    points = np.asarray(points, dtype=np.float64)
    link1 = float(effector.skeleton.L1)
    link2 = float(effector.skeleton.L2)
    x = points[:, 0]
    y = points[:, 1]
    cosine_elbow = (
        x * x + y * y - link1 * link1 - link2 * link2
    ) / (2.0 * link1 * link2)
    reachable = (cosine_elbow >= -1.0) & (cosine_elbow <= 1.0)
    elbow = np.arccos(np.clip(cosine_elbow, -1.0, 1.0))
    shoulder = np.arctan2(y, x) - np.arctan2(
        link2 * np.sin(elbow), link1 + link2 * np.cos(elbow)
    )
    shoulder = np.where(shoulder < 0.0, shoulder + 2.0 * math.pi, shoulder)
    return np.column_stack((shoulder, elbow)), reachable


def joint_states_for_cartesian_starts(points: np.ndarray, effector) -> np.ndarray:
    positions, reachable = inverse_kinematics(points, effector)
    if not bool(reachable.all()):
        raise ValueError("a component start is outside the MotorNet radial workspace")
    lower = np.asarray(effector.pos_lower_bound, dtype=np.float64)
    upper = np.asarray(effector.pos_upper_bound, dtype=np.float64)
    if np.any(positions < lower) or np.any(positions > upper):
        raise ValueError("a component start is outside the MotorNet joint limits")
    return np.column_stack((positions, np.zeros_like(positions)))
