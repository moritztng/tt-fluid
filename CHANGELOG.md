# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- `--lsplit 1` at lowering: a vector that holds an L1 slot across a solver loop that does not use it is parked in DRAM
  for the loop.
- `--tabdense N` at lowering: gathers that read one L1 vector use a dense 16-bit index table (6.4 cycles a value on
  Blackhole against 12.1). `--span 1|2`: a gather from a DRAM-homed vector reads an L1 copy of it through those tables.
  Both keep the state bit-identical and are off by default; `--tscheck` decodes every dense table back to its map
  (`tests/test_dense_span.py`). `ttfluid.tools.gbench`, `spancensus` and `regcensus` measure the gather loop forms and
  count what the tables reach.
- Images record the residue counts `--bpatch` needs, so a run from images chooses it by itself, as it already sized
  the gather cache.
- Packed GPU baseline: `bench/nvidia/pack_plan.sh` runs several SPUMA cases at once on one GPU under MPS. The measured
  H100 cells (1, 2, 4 and 8 cases per GPU) and their raw data are in `results/nvidia/h100-packed/`, and
  `bench/per_dollar.py` prices them.

### Changed
- HPC_motorbike Small on two Blackhole chips: 800 iterations take about 1030 s of device time with `--l1 2 --lsplit 1`
  and the automatic cache and patch settings, 1.18 times faster on the device than `--l1 1 --gcache 14`, with a
  bit-identical state after 100 iterations. The tutorial's Part 2 carries the new lowering and run lines.
- Image upload and the state transfer read only the bytes an image stores. Host time around a Small run fell from
  65 s to 18 s and its peak host memory from 29 GB to 11 GB.
- Lowering HPC_motorbike Small needs about 90 GB of host RAM (the docs said 220 GB).
- A run whose residuals become NaN or infinite ends in that iteration: `run_mesh` prints `DIVERGED` with the
  iteration and the fields, saves the state and exits with status 4 (with `--cases`, after the last case). Before,
  every linear solve of the remaining iterations ran to its cap.

### Fixed
- HPC_motorbike Small could go NaN in single precision: in iteration 337 of one of nine start states on four
  Blackhole chips, and after about 2,200 iterations on two. omega was floored at OpenFOAM's double-precision `SMALL`
  (1e-15). In fp32 it is now floored at 1e-6, the value OpenFOAM's own single-precision build uses; `omegaMin` and
  `kMin` in the RAS dictionary override the defaults. The new floor moves the rounding of every fp32 run from its
  first iterations, so earlier final states are not reproduced bit for bit
  ([accuracy](docs/accuracy.md#a-known-behaviour-of-the-case)). Eight runs on the new floor, the start that went NaN
  included, all end and pool to -5.7 +- 4.9 drag counts against OpenFOAM fp64 (bar 12.5); seven runs on the old floor
  read -3.3 +- 4.9.
- PBiCGStab's breakdown test compared against 1e-300, which is zero in fp32, so it could never stop a solve. It now
  uses the smallest magnitude of the precision.

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
