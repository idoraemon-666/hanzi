"""MotorNet environments for isolated Hanzi components and frozen characters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch as th
from motornet import environment as env

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.geometry import (
    ComponentTrajectory,
    GeometryConfig,
    MoveCondition,
    StrokeCondition,
    build_component_trajectory,
    characters,
    cue_scale,
    load_geometry_config,
)
from hanzi_writing.motornet_support import (
    baseline_anchor,
    joint_states_for_cartesian_starts,
)


DEFAULT_GEOMETRY_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "hanzi_stroke_temporal_composition_geometry.json"
)


class _HanziEnvironment(env.Environment):
    def __init__(
        self,
        *args: Any,
        geometry_config_path: str | Path = DEFAULT_GEOMETRY_CONFIG,
        **kwargs: Any,
    ) -> None:
        self.geometry_config_path = Path(geometry_config_path)
        self.geometry_config = load_geometry_config(self.geometry_config_path)
        super().__init__(*args, **kwargs)
        if self.action_frame_stacking != 0:
            raise ValueError("Hanzi environments require action_frame_stacking=0")
        self.obs_noise[: self.skeleton.space_dim] = [0.0] * self.skeleton.space_dim
        self.dt = self.geometry_config.dt_seconds
        self.anchor_m = baseline_anchor(self.effector)

    def _build_spaces(self) -> None:
        """Declare the fixed spaces without MotorNet 0.2 virtual-reset dispatch."""

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.effector.n_muscles,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
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
        if obs.shape[-1] != 28:
            raise RuntimeError(f"Hanzi observation must have 28 features, got {obs.shape[-1]}")
        if not deterministic:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        return obs if self.differentiable else self.detach(obs)

    def step(
        self, t: int, action: th.Tensor | np.ndarray, **kwargs: Any
    ) -> tuple[th.Tensor | np.ndarray, Any, bool, dict[str, Any]]:
        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32)
        action = action.to(self.device)
        self.effector.step(action, **kwargs)
        obs = self.get_obs(t, action=action)
        reward = None if self.differentiable else np.zeros((action.shape[0], 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self._target_at(t)
        info = {
            "states": self._maybe_detach_states(),
            "action": action if self.differentiable else self.detach(action),
            "noisy action": action if self.differentiable else self.detach(action),
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }
        return obs, reward, terminated, info

    def _reset_effector(self, local_starts: np.ndarray) -> tuple[th.Tensor, th.Tensor]:
        world_starts = np.asarray(local_starts, dtype=np.float64) + self.anchor_m
        joint_states = joint_states_for_cartesian_starts(world_starts, self.effector)
        joint_tensor = th.as_tensor(joint_states, dtype=th.float32, device=self.device)
        self.initial_pos = joint_tensor.clone()
        self.effector.reset(
            options={"batch_size": len(joint_states), "joint_state": joint_tensor}
        )
        fingertip = self.joint2cartesian(joint_tensor).chunk(2, dim=-1)[0]
        return joint_tensor, fingertip

    def _initialize_observation_buffers(
        self, batch_size: int, *, deterministic: bool
    ) -> tuple[Any, dict[str, Any]]:
        action = th.zeros(
            (batch_size, self.action_space.shape[0]),
            dtype=th.float32,
            device=self.device,
        )
        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking
        self.hidden_goal = self._target_at(0)
        obs = self.get_obs(0, deterministic=deterministic)
        output_action = action if self.differentiable else self.detach(action)
        info = {
            "states": self._maybe_detach_states(),
            "action": output_action,
            "noisy action": output_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }
        return obs, info

    def _target_at(self, t: int) -> th.Tensor:
        return self.traj[:, t]


class HanziComponentEnv(_HanziEnvironment):
    """One isolated stroke or move, with no complete-character sequences."""

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
        conditions = options.get("conditions")
        if not isinstance(conditions, Sequence) or not conditions:
            raise ValueError("options.conditions must be a non-empty condition sequence")
        if not all(isinstance(item, (StrokeCondition, MoveCondition)) for item in conditions):
            raise TypeError("conditions must contain StrokeCondition or MoveCondition values")
        speed_name = str(options.get("speed_name"))
        delay_steps = int(options.get("delay_steps"))
        deterministic = bool(options.get("deterministic", False))
        if delay_steps not in self.geometry_config.delay_steps:
            raise ValueError("delay_steps is not in the configured delay grid")

        normalizer = cue_scale(self.geometry_config)
        trajectories = [
            build_component_trajectory(condition, speed_name, normalizer)
            for condition in conditions
        ]
        self._require_batch_compatible(trajectories)
        local = np.stack([trajectory.points_m for trajectory in trajectories])
        self._reset_effector(local[:, 0])
        self.traj = th.as_tensor(
            local + self.anchor_m,
            dtype=th.float32,
            device=self.device,
        )
        self.conditions = tuple(conditions)
        self.component_trajectories = tuple(trajectories)
        self.delay_time = delay_steps
        self.movement_intervals = trajectories[0].movement_intervals
        self._build_trial_inputs(trajectories, delay_steps)
        return self._initialize_observation_buffers(len(conditions), deterministic=deterministic)

    @staticmethod
    def _require_batch_compatible(trajectories: Sequence[ComponentTrajectory]) -> None:
        first = trajectories[0]
        for trajectory in trajectories[1:]:
            if (
                trajectory.rule != first.rule
                or trajectory.speed_name != first.speed_name
                or trajectory.movement_intervals != first.movement_intervals
            ):
                raise ValueError("one batch must share rule, speed, and movement duration")

    def _build_trial_inputs(
        self, trajectories: Sequence[ComponentTrajectory], delay_steps: int
    ) -> None:
        config = self.geometry_config
        movement_start = config.stable_steps + delay_steps
        movement_end = movement_start + trajectories[0].movement_intervals + 1
        hold_end = movement_end + config.hold_steps
        self.epoch_bounds = {
            "stable": (0, config.stable_steps),
            "delay": (config.stable_steps, movement_start),
            "movement": (movement_start, movement_end),
            "hold": (movement_end, hold_end),
        }
        self.max_ep_duration = hold_end - 1
        batch_size = len(trajectories)
        self.rule_input = th.zeros(
            (batch_size, hold_end, authority.RULE_DIM),
            dtype=th.float32,
            device=self.device,
        )
        self.rule_input[:, :, authority.RULE_INDEX[trajectories[0].rule]] = 1.0
        self.speed_scalar = th.zeros(
            (batch_size, hold_end, 1), dtype=th.float32, device=self.device
        )
        self.speed_scalar[:, config.stable_steps :, 0] = trajectories[0].speed_scalar
        self.go_cue = th.zeros_like(self.speed_scalar)
        self.go_cue[:, movement_start:movement_end, 0] = 1.0
        self.vis_inp = th.zeros(
            (batch_size, hold_end, 2), dtype=th.float32, device=self.device
        )
        if trajectories[0].rule == "move":
            cues = th.as_tensor(
                np.stack([trajectory.spatial_cue for trajectory in trajectories]),
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


class HanziCharacterEnv(_HanziEnvironment):
    """A complete frozen-validation schedule; never used by the training sampler."""

    def reset(
        self,
        *,
        testing: bool = True,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if not testing:
            raise ValueError("complete characters are validation-only")
        self._set_generator(seed=seed)
        options = {} if options is None else options
        character_name = str(options.get("character"))
        speed_name = str(options.get("speed_name", "medium"))
        deterministic = bool(options.get("deterministic", True))
        available = characters(self.geometry_config)
        if character_name not in available:
            raise KeyError(f"unknown validation character: {character_name}")
        schedule = authority.assemble_character_schedule(
            available[character_name],
            speed_name=speed_name,
            stable_steps=self.geometry_config.stable_steps,
            prepare_steps=self.geometry_config.prepare_steps,
            final_hold_steps=self.geometry_config.hold_steps,
            cue_normalizer_m=cue_scale(self.geometry_config),
        )
        local_target = np.asarray(schedule["target_xy_m"], dtype=np.float64)
        self._reset_effector(local_target[:1])
        self.traj = th.as_tensor(
            (local_target + self.anchor_m)[None, :, :],
            dtype=th.float32,
            device=self.device,
        )
        self.rule_input = th.as_tensor(
            schedule["rule_input"][None, :, :], dtype=th.float32, device=self.device
        )
        self.speed_scalar = th.as_tensor(
            schedule["speed_scalar"][None, :, :], dtype=th.float32, device=self.device
        )
        self.go_cue = th.as_tensor(
            schedule["go_cue"][None, :, :], dtype=th.float32, device=self.device
        )
        self.vis_inp = th.as_tensor(
            schedule["spatial_goal_cue"][None, :, :],
            dtype=th.float32,
            device=self.device,
        )
        self.writing_mask = th.as_tensor(
            schedule["writing_mask"], dtype=th.bool, device=self.device
        )
        self.phase = tuple(schedule["phase"])
        self.segments = tuple(schedule["segments"])
        self.character_name = character_name
        self.speed_name = speed_name
        self.max_ep_duration = self.traj.shape[1] - 1
        self.epoch_bounds = {
            "stable": (0, self.geometry_config.stable_steps),
            "hold": (
                self.traj.shape[1] - self.geometry_config.hold_steps,
                self.traj.shape[1],
            ),
        }
        return self._initialize_observation_buffers(1, deterministic=deterministic)
