# Public Release Contents (16 August 2026)

## Included

- current Python source, scripts and automated tests;
- versioned experiment configurations and protocols;
- the updated research direction and intended-behaviour contract;
- compact development summaries, endpoint tables, QA records and figures;
- video indexes containing condition IDs, seeds, checkpoint, file name and
  SHA-256 digest;
- the editable team progress presentation.

## Intentionally excluded

- Conda environments and package caches;
- trained PPO model archives;
- compressed step-level trajectories and full training logs;
- complete MP4 panels;
- interrupted or recovery run directories;
- private planning context and machine-specific absolute paths.

The exclusions keep the repository small enough for review and avoid treating
videos or evaluation episodes as independent statistical replications. Full
generated evidence remains in the local research workspace and can be
reproduced from the committed code, configurations and documented seeds.

## Evidence status

The 16 August result package is development evidence. It is suitable for
explaining the mechanism, rejecting tested candidates and freezing a later
formal protocol. It is not a held-out confirmation result.

The historical formal-v1 material is retained for provenance. Its earlier
`ctrl_cost_weight` sweep is not the current central research design.
