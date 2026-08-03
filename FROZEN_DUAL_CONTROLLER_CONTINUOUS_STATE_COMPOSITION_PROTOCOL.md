# Frozen dual-controller continuous physical-state composition protocol

This protocol defines the first complete-character inference experiment that
passes the full MotorNet physical state across Stroke/Move trial boundaries.
It is an inference-only comparison against the endpoint-preserving,
dynamics-reset baseline. It performs no training and changes no checkpoint.

## Frozen controllers

- Stroke: joint-gradient, `onset_window`, `best_macro_checkpoint.pt`.
- Move: joint-gradient, `onset_window`, `best_macro_checkpoint.pt`.
- Both policies run in evaluation mode with gradients and network noise disabled.
- Controller `x/h` states are reset to zero at every trial boundary.

## Character schedule

The rule schedule is unchanged: `mu` contains 7 complete trials, `jiang`
contains 11, and `ke` contains 9. Each character starts with one standard
MotorNet reset at its first canonical Stroke start. There are no MotorNet
resets across the remaining 24 within-character Stroke/Move boundaries.

Every component retains `stable=25`, `delay=50`, its rule-specific movement
interval, and `hold=25`.

## Continuous physical-state boundary

Stroke and Move environments share one MotorNet effector instance. At every
within-character boundary, the next environment configures its rule, cue,
canonical target, epochs, and fresh sensory buffers without calling
`effector.reset` or replacing any effector state.

The following physical states must be bitwise identical before and after the
trial switch, prior to the next controller action:

- `joint`: joint positions and velocities;
- `cartesian`: Cartesian positions and velocities;
- `muscle`: all MotorNet muscle states;
- `geometry`: musculotendon geometry states.

The next rule's canonical target and Move absolute target cue are not
translated. Proprioception and vision buffers are repopulated from the current
continuous body state. Thus physical dynamics continue, while controller
memory and sensory history do not.

## Rendering and interpretation

Every complete trial is rendered from its physical start state through all
`stable`, `delay`, `movement`, and `hold` steps. Complete Stroke trials use
solid lines and complete Move trials use dashed lines. No endpoint snapping,
spatial translation, smoothing, or other trajectory postprocessing is allowed.

The controllers were trained on independently reset trials, so inherited
velocity and muscle state are out of their training distribution. Instability
is a valid outcome. This deterministic single-seed diagnostic defines no
behavioral or aesthetic pass threshold.
