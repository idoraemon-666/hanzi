from __future__ import annotations

import unittest

import numpy as np

from hanzi_writing import hanzi_geometry_final as authority
from hanzi_writing.geometry import (
    ACTIVE_RULES,
    checkpoint_stroke_groups,
    exact_rule_starts,
    load_geometry_config,
    move_conditions,
    stroke_conditions,
    training_conditions_by_rule,
)


GEOMETRY_CONFIG = "configurations/hanzi_stroke_temporal_composition_geometry.json"


class HanziGeometryTests(unittest.TestCase):
    def setUp(self):
        self.config = load_geometry_config(GEOMETRY_CONFIG)

    def test_authority_schedule_and_hold_go_cue(self):
        report = authority.run_self_test()
        self.assertTrue(report["overall_passed"], report["failures"])
        characters, _ = authority.physical_characters()
        normalizer = authority.cue_scale_m(characters)
        for character in characters.values():
            schedule = authority.assemble_character_schedule(
                character, cue_normalizer_m=normalizer
            )
            phase = np.asarray(schedule["phase"])
            movement = np.asarray(["movement" in value for value in phase])
            np.testing.assert_allclose(schedule["go_cue"][movement], 1.0)
            np.testing.assert_allclose(schedule["go_cue"][~movement], 0.0)

    def test_training_sampler_covers_nine_rules_without_complete_characters(self):
        grouped = training_conditions_by_rule(self.config)
        self.assertEqual(tuple(grouped), ACTIVE_RULES)
        self.assertTrue(all(grouped[rule] for rule in ACTIVE_RULES))
        self.assertFalse(any(hasattr(condition, "strokes") for values in grouped.values() for condition in values))

    def test_stroke_start_jitter_is_fixed_and_bounded(self):
        first = stroke_conditions(self.config, include_jitter=True)
        second = stroke_conditions(self.config, include_jitter=True)
        self.assertEqual([row.condition_id for row in first], [row.condition_id for row in second])
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.start_xy_m, right.start_xy_m)
        all_exact = stroke_conditions(self.config, include_jitter=False)
        exact_ids = {condition.condition_id for condition in all_exact}
        self.assertTrue(exact_ids)
        self.assertTrue(any(condition.variant.startswith("jitter_") for condition in first))
        self.assertEqual(set(exact_rule_starts(self.config)), set(ACTIVE_RULES[:-1]))
        characters, _ = authority.physical_characters()
        points = np.concatenate(
            [stroke.points for character in characters.values() for stroke in character.strokes]
        )
        bound = np.ptp(points, axis=0) * self.config.stroke_jitter_fraction
        self.assertEqual(len(first) % 5, 0)
        for index in range(0, len(first), 5):
            exact = first[index]
            self.assertEqual(exact.variant, "exact")
            for jittered in first[index + 1 : index + 5]:
                self.assertTrue(jittered.variant.startswith("jitter_"))
                self.assertTrue(np.all(np.abs(jittered.start_xy_m - exact.start_xy_m) <= bound))

    def test_checkpoint_grid_is_exact_and_has_81_rollout_groups(self):
        stroke_groups = checkpoint_stroke_groups(self.config)
        moves = move_conditions(self.config, include_jitter=False)
        self.assertEqual(len(stroke_groups), 15)
        self.assertEqual(len(moves), 12)
        self.assertEqual((len(stroke_groups) + len(moves)) * 3, 81)
        self.assertTrue(all(condition.variant == "exact" for group in stroke_groups for condition in group))
        self.assertTrue(all(condition.variant == "exact" for condition in moves))
        self.assertEqual(len(move_conditions(self.config, include_jitter=True)), 60)


if __name__ == "__main__":
    unittest.main()
