# Results Sharing

`development_20260816/` contains lightweight, auditable summaries used in the
team update: selected CSV/JSON result tables, compact figures, data-quality
records and video indexes.

The following generated material is intentionally excluded from Git:

- trained PPO model archives;
- compressed step-level trajectories;
- TensorBoard or monitor logs;
- complete MP4 panels;
- interrupted run directories;
- caches and local rendering output.

These exclusions keep the public repository reviewable and avoid publishing
machine-specific paths or treating videos as additional statistical
replications. The committed video indexes identify the prespecified condition,
training seed, evaluation seed and checkpoint; their `video_path` values are
sanitised to filenames rather than local absolute paths.

All committed results are development evidence. The independent replication
unit is a trained policy produced by a training seed, not an evaluation episode
or video.
