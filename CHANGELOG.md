# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/).

## [0.1.0] - 2026-10-19

First public release.

### Added
- `ttfluid` package and CLI: `ttfluid run` (device, after an emulator reference) and `ttfluid check` (CPU emulator only)
  for OpenFOAM simpleFoam cases with k-omega SST.
- numpy SimpleFoam (SIMPLE/SIMPLEC, k-omega SST) with a face-based and a device-layout (ELL) operator backend.
- Resident machine: tracing, RCB partition over chips and cores, scheduling, fusion, encoding, CPU emulator.
- Device runner and persistent kernels for Blackhole (p150, p300) and Wormhole, 1 to 32 chips, ttnn 0.68.0 (verified for this release on a p300, 1 and 2 chips).
- Large-case route: per-chip images, starting states from OpenFOAM time directories.
- OpenFOAM's pitzDaily tutorial, ready to run.
- CPU test suite against OpenFOAM v2412 references; CI with ruff, pytest and gitleaks.
- `bench/per_dollar.py` and the GPU baseline scripts (SPUMA build, run, bound() fix, Cd graders), with the raw data
  behind every published number in `results/`.
- Documentation: getting started, tutorial, architecture, performance methodology, accuracy, FAQ.

### Fixed before release (found by the clean-clone device check)
- `pip install -e '.[tenstorrent]'` now installs torch: the ttnn 0.68.0 wheel does not depend on it, and the device
  runner needs it.
- `ttfluid run --chips 2` on a p300 uses Ethernet links 0 and 1; the Wormhole default asked for link 3 and failed.
- pitzDaily did not converge on the device: the register allocator could place a neighbour gather's result on the
  tile it was still reading from, which the emulator cannot see. A gather's result now never shares a tile with a
  source, and lowering asserts it.
- A case whose `turbulenceProperties` selects a model other than kOmegaSST is refused; it used to run as SST.

[0.1.0]: https://github.com/moritztng/tt-fluid/releases/tag/v0.1.0
