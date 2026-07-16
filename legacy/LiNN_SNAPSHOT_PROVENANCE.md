# LiNN legacy snapshot provenance

## Source

- Repository: `dreamsproxy/LiNN`
- Repository ID: `843133979`
- Source branch at inspection: `main`
- Frozen commit: `6e304b3dff413e892c20dca5b6e55653467c07cc`
- Commit message: `Create connectivity.py`
- Commit timestamp: `2025-05-19T18:59:31Z`

## Destination

- Repository: `dreamsproxy/LiSNN`
- Repository ID: `499461389`
- Path: `legacy/LiNN_snapshot`
- Representation: pinned Git submodule/gitlink
- Migration branch: `merge/linn-snapshot-import`

## Why a submodule was selected

A pinned submodule preserves the donor repository at an exact commit without rewriting source files, flattening its history, or duplicating its binary assets into LiSNN's object database. It also gives the strongest isolation boundary: LiNN remains independently executable and independently versioned, while LiSNN records the precise donor revision.

This replaces the earlier tentative plan to copy only selected text files. The submodule contains the entire donor tree at the frozen commit, including its historical binary assets, but those assets remain stored in the LiNN repository rather than being duplicated into LiSNN.

## Frozen donor tree summary

The frozen revision includes the following major groups:

- Root Oja recurrent network and data-loading experiments.
- `0/space-time-constant.py`.
- `Emergence/` spatial reservoir, cell, connectivity, and visualization experiments.
- `Experiments/` LIF, memory, mycelium, vision, wave, logistic-map, and time-constant experiments.
- `TCLNN Hybrid.py`.
- Historical small images and the `The Fool.wav` audio asset.

No algorithm, import path, filename, generated asset, or dependency declaration was changed by this migration.

## Update rule

The gitlink must not be advanced to a newer LiNN commit without a separately reviewed migration decision and an updated provenance/validation report.
