# Known Issues and Open Decisions

## Scientific blockers

1. A specified natural or task-appropriate gait has not yet been frozen.
2. Ant-v5 leg/contact labels, gait phase and duty-factor definitions require
   tested operational mapping.
3. Existing movement-quality diagnostics do not by themselves establish a
   biologically natural gait.
4. No final V2 reward-shaping or external-constraint candidate is frozen.
5. No V2 held-out formal comparison has been authorised.

## Evidence limitations

1. V1 coefficient runs are retrospective exploratory evidence.
2. The 16 August summaries are development evidence and informed later design.
3. Public video indexes do not include the complete MP4 evidence.
4. Training seed, not evaluation episode or checkpoint, is the independent
   replication unit.
5. Public protocols can precede local execution; protocol presence is not a
   result claim.

## Engineering and provenance

1. Large trained models, compressed trajectories, logs and videos are stored
   outside Git and require a separate SHA-256-verified handover.
2. Existing executable paths are intentionally retained rather than moved into
   `current/` or `legacy/`.
3. Historical formal-v1 path-length and path-efficiency fields include a known
   legacy logging defect; corrected later metrics must not be backfilled into
   the raw files.
4. MuJoCo contact-force diagnostics are simulator quantities, not calibrated
   physical safety measurements.

Resolve an item through a versioned protocol, code/test change and recorded
verification. Do not silently edit historical outputs.
