# Dual fixed-rule joint-gradient protocol

This document freezes the user-approved companion experiment to
`dual_fixed_rule_loss_comparison_v1`. The existing six batch-1 workers remain
unchanged. This companion starts six additional workers in an independent
repository and output tree.

## Experimental question

For each Stroke or Move optimizer step, evaluate every rule exactly once,
scale every complete per-rule objective by `1 / rule_count`, accumulate those
gradients, clip the resulting mean gradient once, and call Adam once. This
tests whether simultaneous task-balanced gradients reduce the interference of
the original one-rule-per-step round-robin schedule.

The three loss arms remain `baseline`, `full_trial`, and `onset_window`. Model
architecture, seed, targets, timing, regularization, learning rates, validation
objective, and checkpoint selection are unchanged.

## Effective task batches

| Model | Per-rule microbatch | Rules per optimizer step | Effective task batch |
|---|---:|---:|---:|
| Stroke | 1 | 15 | all 15 Stroke rules |
| Move | 1 | 12 | all 12 Move rules |

Rules have different trial lengths, so the implementation performs the
forward/backward passes sequentially and accumulates their scaled gradients.
It does not pad unlike trajectories into one tensor. No optimizer step occurs
between rules in the same effective task batch.

The per-rule objective includes the selected position objective plus all
existing regularizers. Averaging the full objective means that the shared
weight regularizer is neither multiplied by the rule count nor weakened.

## Matched exposure and learning-rate schedule

Every rule is exposed exactly once per optimizer step. Each model therefore
uses 8000 optimizer steps and 8000 exposures per rule:

| Model | Phase 1 | Phase 2 | Optimizer steps | Exposures/rule | Total rule rollouts |
|---|---:|---:|---:|---:|---:|
| Stroke | 6000 at `1e-3` | 2000 at `1e-4` | 8000 | 8000 | 120000 |
| Move | 6000 at `1e-3` | 2000 at `1e-4` | 8000 | 8000 | 96000 |

Thus the companion and round-robin experiments perform the same number of
forward/backward rule exposures. They intentionally differ in Adam step count:
the scientific intervention is one Adam update per rule versus one Adam update
after the mean gradient of a complete rule set.

The Adam learning rates stay at `1e-3` and `1e-4`; no batch-size scaling is
applied. The accumulated gradient is clipped once at norm `1.0`.

## Validation and checkpoints

Stroke validates every 40 joint optimizer steps and Move every 50. These are
respectively equivalent to 600 rule exposures in the original 15-rule and
12-rule schedules, preserving 200 Stroke and 160 Move scheduled validations.
Every validation still evaluates all rules without network or observation
noise and uses the common phase-normalized position metric.

Each worker retains macro-best, worst-rule-best, final, and continuation
checkpoints under the joint-gradient variant identity. Joint-gradient and
round-robin checkpoints are intentionally incompatible.

## Server isolation

The companion must not update or reuse the repository from which the original
six workers are running. It uses a separate clean repository, a separate run
directory, and the formal launcher:

```text
server/run_hanzi_dual_fixed_rule_joint_gradient.sh
```

Both launchers use the same frozen CPU environment. With one CPU thread per
worker, the combined experiment contains twelve training workers. No complete
character or chained-controller training is included.
