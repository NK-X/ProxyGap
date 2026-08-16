# Body-dynamics replication gate V1

**Status:** frozen authorised final development replication; formal launch
prohibited.

This protocol operationalises
`docs/FUTURE_TESTING_DIRECTION_20260817.md`. It compares the completed
ordinary-exploration baseline and body-dynamics reward packages using three new
development training seeds. The body penalty, all shared reward terms, PPO
configuration, training budget and evaluation design remain fixed.

The executable record is
`configs/body_dynamics_replication_v1_20260817.json`. The existing body-matrix
runner accepts this declared two-condition replication design and refuses an
unknown condition set, overlapping formal seeds, an inconsistent evaluation
seed list or a non-frozen status.

The experiment estimates whether a previously observed mechanism replicates.
It is not a fresh reward search, a natural-gait test or a held-out formal
comparison. Outputs are local by default and must not be added to the public
repository before evidence and privacy review.
