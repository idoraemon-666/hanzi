# Dual fixed-rule Stroke/Move RNN protocol

This document records the user-approved design implemented by
`dual_fixed_rule_loss_comparison_v1`. It is isolated from every existing
28-D canonical path.

## Experimental question

Train two independent multitask controllers:

- one Stroke RNN with 15 fixed one-hot rules;
- one Move RNN with 12 fixed one-hot rules.

Each controller is trained from a matched fresh seed under three position-loss
arms: `baseline`, `full_trial`, and `onset_window`. The six workers are the
formal comparison. They do not train a complete character or chain the two
controllers.

Every one-hot maps to exactly one target trajectory, start, direction,
movement duration, and spatial cue. Therefore this experiment tests fixed-rule
multitask learning, not length, direction, speed, or delay generalization.

## Fixed observation contracts

Stroke RNN, 33 dimensions:

```text
[0:15]   rule one-hot
[15:16]  fixed speed/duration scalar = 0.0
[16:17]  go cue
[17:19]  spatial cue = [0, 0]
[19:21]  fingertip vision
[21:33]  muscle proprioception
```

Move RNN, 30 dimensions:

```text
[0:12]   rule one-hot
[12:13]  fixed speed/duration scalar = 0.0
[13:14]  go cue
[14:16]  absolute target spatial cue
[16:18]  fingertip vision
[18:30]  muscle proprioception
```

Both retain the existing 256-unit softplus RNN and six muscle outputs. Old
28-D checkpoints are intentionally incompatible and are rejected before state
loading.

## Fixed trial timing

```text
dt             = 0.01 s
stable_steps   = 25
delay_steps    = 50
movement_steps = rule-specific table below
hold_steps     = 25
batch_size     = 1
```

The constant slow scalar is an input-compatibility cue; it is not interpreted
as a common physical speed. No direction, speed, delay, start, or target jitter
is sampled.

## Stroke rules and movement intervals

| Index | Rule | Intervals | Timing basis |
|---:|---|---:|---|
| 0 | `long_heng_ke_0` | 150 | existing canonical |
| 1 | `long_heng_jiang_5` | 126 | canonical heng distance/interval |
| 2 | `medium_heng_mu_0` | 106 | canonical heng distance/interval |
| 3 | `medium_heng_jiang_3` | 75 | canonical heng distance/interval |
| 4 | `short_heng_ke_3` | 45 | canonical heng distance/interval |
| 5 | `long_shu_mu_1` | 150 | existing canonical |
| 6 | `medium_shu_jiang_4` | 56 | canonical shu distance/interval |
| 7 | `short_shu_ke_1` | 39 | canonical shu distance/interval |
| 8 | `pie_mu_2` | 150 | existing canonical |
| 9 | `na_mu_3` | 150 | existing canonical |
| 10 | `dian_jiang_0` | 150 | existing canonical |
| 11 | `dian_jiang_1` | 129 | canonical dian distance/interval |
| 12 | `ti_jiang_2` | 150 | existing canonical |
| 13 | `hengzhe_ke_2` | 200 | existing canonical rounded target |
| 14 | `shugou_ke_4` | 200 | existing canonical rounded target |

The eight existing canonical representatives are preserved point-for-point.
New occurrences use
`ceil(150 * new_length / same_family_canonical_length)`.

## Move rules and movement intervals

The common reference is `0.8502257335 mm/interval`, the median measured
distance per interval of the six successfully fitted canonical simple strokes.

| Index | Rule | Intervals |
|---:|---|---:|
| 0 | `mu_move_0` | 125 |
| 1 | `mu_move_1` | 161 |
| 2 | `mu_move_2` | 147 |
| 3 | `jiang_move_0` | 61 |
| 4 | `jiang_move_1` | 121 |
| 5 | `jiang_move_2` | 56 |
| 6 | `jiang_move_3` | 73 |
| 7 | `jiang_move_4` | 79 |
| 8 | `ke_move_0` | 199 |
| 9 | `ke_move_1` | 65 |
| 10 | `ke_move_2` | 42 |
| 11 | `ke_move_3` | 94 |

## Training schedule

Rules use deterministic fixed round-robin scheduling. Every phase boundary,
log interval, and validation interval is an exact whole number of cycles.

| Model | Phase 1 | Phase 2 | Total | Updates/rule |
|---|---:|---:|---:|---:|
| Stroke | 90k at `1e-3` | 30k at `1e-4` | 120k | 8000 |
| Move | 72k at `1e-3` | 24k at `1e-4` | 96k | 8000 |

Validation runs every 600 completed updates on every rule, with network and
observation noise disabled. All three loss arms are evaluated by the same
phase-normalized position metric.

## Checkpoint selection

Each worker retains:

- `best_macro_checkpoint.pt`: minimizes `(macro mean, worst rule)`;
- `best_worst_checkpoint.pt`: minimizes `(worst rule, macro mean)`;
- `final_checkpoint.pt`;
- `continuation_checkpoint.pt` with optimizer and RNG state.

The first two are threshold-free complementary selections. Final diagnostics
evaluate all three candidate model states per worker and report every rule;
compound metrics remain descriptive and are not silently folded into a new
loss.

Checkpoints bind the model kind, loss arm, rule count, input size, rule order,
and condition-manifest SHA-256. Validation is read-only with explicit checks
for policy, optimizer, gradients, global RNG, environment RNG, training mode,
and protected checkpoint hashes.

## Formal server entry

The only formal launcher for this variant is:

```text
server/run_hanzi_dual_fixed_rule_loss_comparison.sh
```

It requires explicit authorization, a clean expected Git HEAD, the frozen
submodule HEAD, the project CPU environment, and six one-thread workers. It
refuses to overwrite any output or archive.
