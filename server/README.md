# Server files

The active branch is `hanzi_stroke_temporal_composition`. Its server files are:

- `create_hanzi_cpu_environment.sh`: creates the isolated CPU environment using
  domestic Conda/PyPI/PyTorch mirrors;
- `run_hanzi_stroke_temporal_composition_audit.sh`: creates/resumes that
  environment, runs all tests and required pre-training audits, and never trains;
- `run_hanzi_stroke_temporal_composition_train_dev42.sh`: separately authorized
  75,000-update training;
- `run_hanzi_stroke_temporal_composition_validate_characters.sh`: separately
  authorized frozen `mu/jiang/ke` validation.
- `export_hanzi_legacy_timing_reference.py`: read-only old-HEAD timing exporter;
- `run_hanzi_fixed_duration_preflight.sh`: fixed-duration target audit, deterministic
  coverage, 100-update technical smoke, and read-only validation benchmark. It
  stops without starting the 10k pilot or either formal training run.

Use the generated upload package and its short `RUN_AUDIT.sh` command instead of
manually reconstructing the repository or pasting a long command sequence.

The older project-2 files below remain intact for provenance and regression use.

This directory contains the fixed CPU environment files, the project-2 audit
runner, and four independently authorized experiment wrappers.

Read `PROJECT_PROTOCOL.md`, `NEXT_SESSION_HANDOFF.md`, and
`AUTODL_SERVER_USAGE_GUIDE.md` before any server transfer or execution.

Project-2 audits use `run_digit_writing_original_protocol2_audit.sh`.
Formal runs use the project-2 wrappers and require the matching
`DIGIT_PROTOCOL2_AUTHORIZED_RUN` exact label.
