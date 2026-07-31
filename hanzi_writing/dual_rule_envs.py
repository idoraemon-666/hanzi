"""Isolated MotorNet environment for the dual fixed-rule RNN experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch as th

from hanzi_writing.dual_rule_protocol import (
    FIXED_DELAY_STEPS,
    FIXED_SPEED_NAME,
    FixedRuleCondition,
    component_trajectory,
    input_size,
    rule_index,
    rule_names,
)
from hanzi_writing.envs import _HanziEnvironment


class DualFixedRuleEnv(_HanziEnvironment):
    """One fixed stroke or move rule per batch, with a model-specific one-hot."""

    def __init__(
        self,
        *args: Any,
        model_kind: str,
        geometry_config_path: str | Path,
        **kwargs: Any,
    ) -> None:
        self.model_kind = model_kind
        self.dual_rule_names = rule_names(model_kind)
        self.dual_rule_index = rule_index(model_kind)
        self.dual_input_size = input_size(model_kind)
        super().__init__(
            *args,
            geometry_config_path=geometry_config_path,
            **kwargs,
        )

    def _build_spaces(self) -> None:
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.effector.n_muscles,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dual_input_size,),
            dtype=np.float32,
        )
        self.action_noise = [self._action_noise] * self.action_space.shape[0]
        self.obs_noise = [self._obs_noise] * self.observation_space.shape[0]

    def get_obs(
        self,
        t: int,
        action: th.Tensor | np.ndarray | None = None,
        deterministic: bool = False,
    ) -> th.Tensor | np.ndarray:
        self.update_obs_buffer(action=action)
        obs = th.cat(
            [
                self.rule_input[:, t],
                self.speed_scalar[:, t],
                self.go_cue[:, t],
                self.vis_inp[:, t],
                self.obs_buffer["vision"][0],
                self.obs_buffer["proprioception"][0],
            ],
            dim=-1,
        )
        if obs.shape[-1] != self.dual_input_size:
            raise RuntimeError(
                f"dual {self.model_kind} observation must have "
                f"{self.dual_input_size} features, got {obs.shape[-1]}"
            )
        if not deterministic:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        return obs if self.differentiable else self.detach(obs)

    def reset(
        self,
        *,
        testing: bool = False,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        del testing
        self._set_generator(seed=seed)
        options = {} if options is None else options
        supplied = options.get("conditions")
        if not isinstance(supplied, Sequence) or not supplied:
            raise ValueError("options.conditions must be a non-empty sequence")
        if not all(isinstance(item, FixedRuleCondition) for item in supplied):
            raise TypeError("dual conditions must contain FixedRuleCondition values")
        conditions = tuple(supplied)
        if any(condition.model_kind != self.model_kind for condition in conditions):
            raise ValueError("condition model kind differs from the environment")
        if options.get("speed_name") != FIXED_SPEED_NAME:
            raise ValueError("dual fixed-rule trials require the frozen slow cue")
        if int(options.get("delay_steps")) != FIXED_DELAY_STEPS:
            raise ValueError("dual fixed-rule trials require delay_steps=50")
        deterministic = bool(options.get("deterministic", False))
        trajectories = tuple(component_trajectory(condition) for condition in conditions)
        self._require_batch_compatible(trajectories)
        local = np.stack([trajectory.points_m for trajectory in trajectories])
        self._reset_effector(local[:, 0])
        self.traj = th.as_tensor(
            local + self.anchor_m,
            dtype=th.float32,
            device=self.device,
        )
        self.conditions = conditions
        self.component_trajectories = trajectories
        self.delay_time = FIXED_DELAY_STEPS
        self.movement_intervals = trajectories[0].movement_intervals
        self.base_movement_intervals = trajectories[0].base_movement_intervals
        self.movement_subphase = trajectories[0].movement_subphase
        self._build_trial_inputs(conditions, FIXED_DELAY_STEPS)
        return self._initialize_observation_buffers(
            len(conditions), deterministic=deterministic
        )

    @staticmethod
    def _require_batch_compatible(trajectories: Sequence[Any]) -> None:
        first = trajectories[0]
        for trajectory in trajectories[1:]:
            if (
                trajectory.rule != first.rule
                or trajectory.speed_name != first.speed_name
                or trajectory.movement_intervals != first.movement_intervals
            ):
                raise ValueError("one dual-RNN batch must contain one fixed rule")

    def _build_trial_inputs(
        self,
        conditions: Sequence[FixedRuleCondition],
        delay_steps: int,
    ) -> None:
        config = self.geometry_config
        movement_start = config.stable_steps + delay_steps
        movement_end = movement_start + conditions[0].movement_intervals + 1
        hold_end = movement_end + config.hold_steps
        self.epoch_bounds = {
            "stable": (0, config.stable_steps),
            "delay": (config.stable_steps, movement_start),
            "movement": (movement_start, movement_end),
            "hold": (movement_end, hold_end),
        }
        self.max_ep_duration = hold_end - 1
        batch_size = len(conditions)
        self.rule_input = th.zeros(
            (batch_size, hold_end, len(self.dual_rule_names)),
            dtype=th.float32,
            device=self.device,
        )
        for batch_index, condition in enumerate(conditions):
            self.rule_input[
                batch_index, :, self.dual_rule_index[condition.rule]
            ] = 1.0
        self.speed_scalar = th.zeros(
            (batch_size, hold_end, 1), dtype=th.float32, device=self.device
        )
        self.go_cue = th.zeros_like(self.speed_scalar)
        self.go_cue[:, movement_start:movement_end, 0] = 1.0
        self.vis_inp = th.zeros(
            (batch_size, hold_end, 2), dtype=th.float32, device=self.device
        )
        if self.model_kind == "move":
            cues = th.as_tensor(
                np.stack([condition.spatial_cue for condition in conditions]),
                dtype=th.float32,
                device=self.device,
            )
            self.vis_inp[:, config.stable_steps:movement_end] = cues[:, None, :]

        target = th.empty(
            (batch_size, hold_end, 2), dtype=th.float32, device=self.device
        )
        target[:, :movement_start] = self.traj[:, :1]
        target[:, movement_start:movement_end] = self.traj
        target[:, movement_end:] = self.traj[:, -1:]
        self.traj = target
