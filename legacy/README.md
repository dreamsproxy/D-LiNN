# Legacy implementations

Code referenced under `legacy/` is preserved for historical comparison, reproducibility, and later component extraction. It is not part of the active LiSNN API and must not be imported by production modules or collected by the default test suite.

## LiNN snapshot

`legacy/LiNN_snapshot` is a Git submodule pinned to the frozen donor revision of `dreamsproxy/LiNN`.

- Donor repository: `dreamsproxy/LiNN`
- Frozen commit: `6e304b3dff413e892c20dca5b6e55653467c07cc`
- Commit message: `Create connectivity.py`
- Destination role: legacy/reference lineage
- Active API status: excluded

LiNN was an independent reimplementation created after the earlier D-LiNN lineage had been forgotten. D-LiNN was later renamed LiSNN. The two repositories therefore represent parallel experimental histories rather than a normal fork relationship.

## Checkout

Because LiNN is private, cloning LiSNN with its legacy snapshot requires GitHub credentials that can read both repositories:

```bash
git clone --recurse-submodules https://github.com/dreamsproxy/LiSNN.git
```

For an existing clone:

```bash
git submodule update --init --recursive
```

The gitlink in LiSNN pins the exact donor commit. Updating the LiNN default branch does not update this snapshot automatically.

## Isolation policy

- Active LiSNN code must not import from `legacy/LiNN_snapshot`.
- Legacy scripts may contain top-level execution, hardcoded paths, incomplete experiments, and generated-output behavior.
- Legacy dependencies are not automatically promoted into LiSNN's primary environment.
- Refactoring or correcting LiNN behavior must occur outside the pinned snapshot and only after a separately approved extraction phase.
