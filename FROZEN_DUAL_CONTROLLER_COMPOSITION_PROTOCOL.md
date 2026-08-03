# Frozen dual-controller character composition protocol

This protocol freezes the first complete-character inference experiment built
from the completed dual fixed-rule RNNs. It performs no training and changes no
checkpoint.

## Frozen controllers

- Stroke: joint-gradient, `onset_window`, `best_macro_checkpoint.pt`.
- Move: joint-gradient, `onset_window`, `best_macro_checkpoint.pt`.
- Both policies run in evaluation mode with gradients and network noise disabled.

## Character schedule

Each character alternates one complete Stroke trial with one complete Move
trial. The final stroke has no following move.

```text
mu:    stroke 0, move 0, stroke 1, move 1, stroke 2, move 2, stroke 3
jiang: stroke 0, move 0, stroke 1, move 1, ..., move 4, stroke 5
ke:    stroke 0, move 0, stroke 1, move 1, ..., move 3, stroke 4
```

Every component retains its frozen `stable=25`, `delay=50`, rule-specific
movement interval, and `hold=25` phases.

## Trial boundary

The actual fingertip endpoint after the complete preceding trial is the reset
position of the next trial. MotorNet is reinitialized at that actual endpoint
with its standard zero-velocity state. The next controller receives fresh zero
`x/h` states, and its feedback buffers are repopulated from the reset physical
state.

The next rule's canonical target and Move absolute target cue are not
translated. Therefore spatial endpoint error is allowed to propagate into the
next task, while velocity and muscle-state carry-over are intentionally
excluded. This most closely matches the independent-trial training contract.

## Rendering and interpretation

Every actual complete-trial trajectory is rendered from its reset state through
all `stable`, `delay`, `movement`, and `hold` steps. The reset state is prepended
to the post-action state sequence, so adjacent trials visibly meet at the
actual endpoint-to-reset boundary. Complete Stroke trials use solid lines and
complete Move trials use dashed lines. No endpoint snapping, spatial
translation, smoothing, or other visual postprocessing is permitted.

The output is a deterministic single-seed composition diagnostic. It does not
define a behavioral pass threshold and does not establish generalization.
