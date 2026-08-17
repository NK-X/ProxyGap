# ProxyGap Handover: Start Here

## Reading order

1. `README.md` - project scope and version map.
2. `STATUS.md` - current scientific and engineering state.
3. `current/README.md` - current V2 workflow.
4. `current/RESEARCH_DIRECTION_V2.md` - next scientific design gate.
5. `docs/V2_FILE_GUIDE_AND_VIDEO_REPRODUCTION_CN.md` - task-oriented V2 file
   map and video requirements for Chinese-speaking collaborators.
6. `handoff/RUN_REGISTRY.csv` - experiment-family provenance.
7. `handoff/DATA_DICTIONARY.md` - data and evidence meanings.
8. `handoff/KNOWN_ISSUES.md` - unresolved problems and claim limits.
9. `docs/REPRODUCIBILITY.md` and `CONTRIBUTING.md` - local setup and change rules.

## Repository versus full evidence bundle

This GitHub repository is the public, reviewable code and protocol record. It
does not contain all trained models, compressed trajectories, complete videos,
recovery folders or local environments. A person receiving the full project
must obtain the separately stored evidence bundle and verify it against a
SHA-256 manifest before analysis.

Absence of a large artifact from Git is not evidence that the run did not
exist. Conversely, the presence of a protocol does not prove that its run was
completed or scientifically successful.

## Minimum local verification

```powershell
conda env create -f environment.yml
conda activate proxygap-ant
python -m pip install -e .
python -m pytest tests
python scripts/build_public_manifest.py --check
```

The public manifest hashes the canonical bytes in the Git index rather than
platform-specific working-tree line endings.

For a new experiment, create a branch and version the scientific protocol,
configuration, seed rules, stopping rules and output schema before launching a
long run. Do not overwrite raw output directories.

## Handover checklist

- confirm the Git commit and version tag;
- verify the public file manifest;
- obtain and hash-check the external evidence bundle;
- identify whether each run is smoke, development, retrospective or held-out;
- preserve reserved training seeds;
- record any environment or command deviation;
- do not interpret videos as independent replications;
- do not merge V1 and V2 results into one formal dataset.
