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
- Device runner and persistent kernels for Blackhole (p150, p300) and Wormhole, 1 to 32 chips, ttnn 0.68.0.
- Large-case route: per-chip images, starting states from OpenFOAM time directories.
- OpenFOAM's pitzDaily tutorial, ready to run.
- CPU test suite against OpenFOAM v2412 references; CI with ruff, pytest and gitleaks.
- `bench/per_dollar.py` and the GPU baseline scripts (SPUMA build, run, bound() fix, Cd graders), with the raw data
  behind every published number in `results/`.
- Documentation: getting started, tutorial, architecture, performance methodology, accuracy, FAQ.

[0.1.0]: https://github.com/moritztng/tt-fluid/releases/tag/v0.1.0
