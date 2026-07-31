from __future__ import annotations

import copy
from pathlib import Path
import unittest

import numpy as np
import torch

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.canonical_protocol import (
    canonical_stroke_conditions,
    canonical_target_trajectory,
)
from hanzi_writing.dual_rule_envs import DualFixedRuleEnv
from hanzi_writing.dual_rule_protocol import (
    FIXED_DELAY_STEPS,
    LOSS_ARMS,
    MOVE_RULES,
    STROKE_RULES,
    condition_manifest,
    conditions,
    input_size,
    rule_names,
)
from hanzi_writing.dual_rule_training import (
    CONFIG_PATH,
    _make_env,
    _position_objective,
    _runtime_hp,
    load_config,
    readonly_validation,
    scheduled_rule,
    validate_checkpoint_identity,
)
from hanzi_writing.training import _make_effector, _rollout
from train import _build_policy


ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = ROOT / "server" / "run_hanzi_dual_fixed_rule_loss_comparison.sh"


class DualFixedRuleRNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.geometry = load_config(CONFIG_PATH)

    def test_configuration_freezes_two_models_and_six_workers(self) -> None:
        self.assertEqual(tuple(self.config["loss_arms"]), LOSS_ARMS)
        self.assertEqual(self.config["training"]["parallel_processes"], 6)
        self.assertEqual(self.config["training"]["batch_size"], 1)
        self.assertEqual(self.config["timing"]["delay_steps"], 50)
        self.assertFalse(self.config["timing"]["direction_augmentation"])
        self.assertEqual(
            self.config["models"]["stroke"],
            {
                "rule_dim": 15,
                "input_size": 33,
                "max_updates": 120000,
                "phase1_updates": 90000,
                "phase2_updates": 30000,
                "log_interval": 150,
            },
        )
        self.assertEqual(
            self.config["models"]["move"],
            {
                "rule_dim": 12,
                "input_size": 30,
                "max_updates": 96000,
                "phase1_updates": 72000,
                "phase2_updates": 24000,
                "log_interval": 120,
            },
        )

    def test_rule_libraries_are_one_condition_per_one_hot(self) -> None:
        stroke = conditions("stroke", self.geometry)
        move = conditions("move", self.geometry)
        self.assertEqual(tuple(item.rule for item in stroke), STROKE_RULES)
        self.assertEqual(tuple(item.rule for item in move), MOVE_RULES)
        self.assertEqual(len({item.condition_id for item in stroke}), 15)
        self.assertEqual(len({item.condition_id for item in move}), 12)
        self.assertEqual(input_size("stroke"), 33)
        self.assertEqual(input_size("move"), 30)

    def test_frozen_movement_intervals_match_the_approved_tables(self) -> None:
        stroke = {item.rule: item.movement_intervals for item in conditions("stroke", self.geometry)}
        move = {item.rule: item.movement_intervals for item in conditions("move", self.geometry)}
        self.assertEqual(
            stroke,
            {
                "long_heng_ke_0": 150,
                "long_heng_jiang_5": 126,
                "medium_heng_mu_0": 106,
                "medium_heng_jiang_3": 75,
                "short_heng_ke_3": 45,
                "long_shu_mu_1": 150,
                "medium_shu_jiang_4": 56,
                "short_shu_ke_1": 39,
                "pie_mu_2": 150,
                "na_mu_3": 150,
                "dian_jiang_0": 150,
                "dian_jiang_1": 129,
                "ti_jiang_2": 150,
                "hengzhe_ke_2": 200,
                "shugou_ke_4": 200,
            },
        )
        self.assertEqual(
            move,
            {
                "mu_move_0": 125,
                "mu_move_1": 161,
                "mu_move_2": 147,
                "jiang_move_0": 61,
                "jiang_move_1": 121,
                "jiang_move_2": 56,
                "jiang_move_3": 73,
                "jiang_move_4": 79,
                "ke_move_0": 199,
                "ke_move_1": 65,
                "ke_move_2": 42,
                "ke_move_3": 94,
            },
        )

    def test_existing_eight_canonical_targets_are_bitwise_preserved(self) -> None:
        old_conditions = canonical_stroke_conditions(self.geometry)
        old = {
            (item.character, item.stroke_index): canonical_target_trajectory(
                item, self.geometry
            )["points_m"]
            for item in old_conditions
        }
        new = {
            (item.source_character, item.source_component_index): item.points_m
            for item in conditions("stroke", self.geometry)
        }
        self.assertEqual(len(old), 8)
        for key, points in old.items():
            self.assertTrue(np.array_equal(points, new[key]), key)

    def test_all_targets_preserve_endpoints_and_have_no_zero_intervals(self) -> None:
        for model_kind in ("stroke", "move"):
            for condition in conditions(model_kind, self.geometry):
                self.assertEqual(
                    len(condition.points_m), condition.movement_intervals + 1
                )
                self.assertTrue(np.isfinite(condition.points_m).all())
                self.assertTrue(
                    np.all(
                        np.linalg.norm(np.diff(condition.points_m, axis=0), axis=1)
                        > 0.0
                    )
                )

    def test_manifest_hash_is_deterministic_and_model_specific(self) -> None:
        first = condition_manifest("stroke", self.geometry)
        second = condition_manifest("stroke", self.geometry)
        move = condition_manifest("move", self.geometry)
        self.assertEqual(first, second)
        self.assertNotEqual(
            first["condition_manifest_sha256"],
            move["condition_manifest_sha256"],
        )
        self.assertEqual(first["rule_names"], list(STROKE_RULES))
        self.assertEqual(move["rule_names"], list(MOVE_RULES))

    def test_round_robin_cycles_and_learning_rate_phases_are_exact(self) -> None:
        for model_kind, phase1, phase2 in (
            ("stroke", 90000, 30000),
            ("move", 72000, 24000),
        ):
            names = STROKE_RULES if model_kind == "stroke" else MOVE_RULES
            self.assertEqual(
                tuple(scheduled_rule(model_kind, update) for update in range(len(names))),
                names,
            )
            self.assertEqual(phase1 // len(names), 6000)
            self.assertEqual(phase2 // len(names), 2000)

    def test_environment_dimensions_one_hot_and_fixed_delay(self) -> None:
        for model_kind, expected_size in (("stroke", 33), ("move", 30)):
            library = conditions(model_kind, self.geometry)
            env = DualFixedRuleEnv(
                effector=_make_effector(),
                model_kind=model_kind,
                geometry_config_path=self.config["geometry_config"],
                action_frame_stacking=0,
            )
            observation, _ = env.reset(
                options={
                    "conditions": (library[0],),
                    "speed_name": "slow",
                    "delay_steps": FIXED_DELAY_STEPS,
                    "deterministic": True,
                }
            )
            self.assertEqual(observation.shape, (1, expected_size))
            one_hot = observation[0, : len(library)].detach().cpu().numpy()
            self.assertEqual(float(one_hot.sum()), 1.0)
            self.assertEqual(float(one_hot[0]), 1.0)
            with self.assertRaisesRegex(ValueError, "delay_steps=50"):
                env.reset(
                    options={
                        "conditions": (library[0],),
                        "speed_name": "slow",
                        "delay_steps": 25,
                        "deterministic": True,
                    }
                )

    def test_checkpoint_contract_rejects_cross_model_and_rule_order(self) -> None:
        stroke_manifest = condition_manifest("stroke", self.geometry)
        checkpoint = {
            "project": "hanzi_stroke_temporal_composition",
            "variant": "dual_fixed_rule_loss_comparison_v1",
            "model_kind": "stroke",
            "loss_arm": "baseline",
            "rule_dim": 15,
            "input_size": 33,
            "rule_names": list(STROKE_RULES),
            "condition_manifest_sha256": stroke_manifest[
                "condition_manifest_sha256"
            ],
            "hp": {"inp_size": 33},
        }
        validate_checkpoint_identity(
            checkpoint, "stroke", "baseline", stroke_manifest
        )
        with self.assertRaisesRegex(ValueError, "identity differs"):
            validate_checkpoint_identity(
                checkpoint,
                "move",
                "baseline",
                condition_manifest("move", self.geometry),
            )
        changed = copy.deepcopy(checkpoint)
        changed["rule_names"] = list(reversed(STROKE_RULES))
        with self.assertRaisesRegex(ValueError, "identity differs"):
            validate_checkpoint_identity(
                changed, "stroke", "baseline", stroke_manifest
            )

    def test_both_model_shapes_backpropagate_and_validate_read_only(self) -> None:
        for model_kind in ("stroke", "move"):
            torch.manual_seed(42)
            hp = _runtime_hp(self.config, model_kind)
            policy = _build_policy(hp, 6, torch.device("cpu"))
            optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
            env = _make_env(self.config, model_kind)
            library = conditions(model_kind, self.geometry)
            for arm in LOSS_ARMS:
                result = _rollout(
                    policy,
                    env,
                    hp,
                    (library[0],),
                    "slow",
                    FIXED_DELAY_STEPS,
                    network_noise=False,
                    deterministic_observation=True,
                    track_gradients=True,
                )
                objective, _ = _position_objective(arm, result)
                optimizer.zero_grad()
                objective.backward()
                gradients = [
                    parameter.grad
                    for parameter in policy.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
                optimizer.step()
            readonly = readonly_validation(
                policy,
                optimizer,
                env,
                hp,
                self.config,
                model_kind,
                library,
            )
            self.assertEqual(
                readonly["validation"]["rule_count"], len(rule_names(model_kind))
            )
            self.assertTrue(all(readonly["read_only_checks"].values()))

    def test_old_rule_authority_and_server_six_worker_gate_remain_explicit(self) -> None:
        self.assertEqual(authority.RULE_DIM, 10)
        script = SERVER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("MODEL_KINDS=(stroke move)", script)
        self.assertIn("ARMS=(baseline full_trial onset_window)", script)
        self.assertIn('test "${#WORKER_PIDS[@]}" -eq 6', script)
        self.assertIn("OMP_NUM_THREADS=1", script)
        self.assertIn("STROKE_UPDATES=120000", script)
        self.assertIn("MOVE_UPDATES=96000", script)
        self.assertIn("FIXED_DELAY_STEPS=50", script)
        self.assertIn("COMPLETE_CHARACTER_STARTED=0", script)


if __name__ == "__main__":
    unittest.main()
