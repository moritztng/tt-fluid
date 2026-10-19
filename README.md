# tt-fluid

[![CI](https://github.com/moritztng/tt-fluid/actions/workflows/ci.yml/badge.svg)](https://github.com/moritztng/tt-fluid/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

**tt-fluid runs OpenFOAM's steady incompressible solver (simpleFoam, k-omega SST) entirely on Tenstorrent chips.** It
reads an ordinary OpenFOAM case directory and runs every SIMPLE iteration on the device: momentum assembly, the
turbulence model, all four linear solves and the flux correction. It writes the fields back as OpenFOAM time directories.
OpenFOAM is not needed at run time. The same program also runs on a CPU emulator, which reproduces the device result bit
for bit and needs no hardware.

On the OpenFOAM HPC benchmark's motorbike case (HPC_motorbike Small: 8.66 M cells, 800 SIMPLE iterations), a $9,999 Tenstorrent QuietBox 2
costs **$0.115 per case, 2.34x less than an 8x H100 SXM server** running the GPU port of OpenFOAM (SPUMA) at its
cheapest documented price, and 3.19x less than 8x B200. The drag coefficient lands 10 drag counts (0.2 %) below OpenFOAM's
fp64 CPU result, inside the reference's own uncertainty, while the eight H100 runs scatter from -4 to +30. Every number in this README is printed by a script in this repository from raw
data in this repository.

![Cost per case and drag coefficient vs OpenFOAM](docs/figures/headline.svg)

## Why it matters

Steady RANS is still most of the CFD that engineering teams run, and GPU ports of OpenFOAM are bound by memory
bandwidth. A Tenstorrent chip is a grid of more than a hundred small cores, each with its own SRAM, linked by a network on chip,
and chips link to each other over Ethernet. tt-fluid keeps the whole iteration resident on that grid: each core owns a
block of cells, exchanges halos with its neighbours directly, and never returns to the host inside a run. The result is
a cheaper box doing the same job, with its accuracy checked against OpenFOAM.

## Results

HPC_motorbike Small (OpenFOAM HPC Technical Committee benchmark, the motorBike tutorial geometry on a finer
mesh), 8,662,309 cells, k-omega SST, 800 SIMPLE iterations from the same
potentialFoam start on both sides. Cost per case uses one formula for every system: `capex / 3 years + wall kW x PUE
1.3 x $0.15/kWh`, then `$/case = batch wall time x $/h / cases in the batch` ([method](docs/performance.md)).

| system | price | cases at once | batch wall | $/case | per $ vs this system | how | raw data |
|---|---|---|---|---|---|---|---|
| Tenstorrent QuietBox 2 (2x p300c, 4 Blackhole chips) | $9,999 | 2 (one per board) | 1310.3 s | **$0.1154** | 1.00x | measured, both boards at once, AICLK 1350 MHz on all 4 chips | [logs](results/tenstorrent/quietbox2-p300x4) |
| 8x H100 SXM (HGX), SPUMA, mixed precision | $249,200 | 8 (one per GPU) | 771.5 s | $0.2704 | **2.34x** | measured, 8 cases at once | [runs](results/nvidia/dgx-h100) |
| 8x B200 (HGX), SPUMA, mixed precision | $384,228 | 8 (one per GPU) | 690.5 s | $0.3678 | **3.19x** | one B200 measured, 8x derived | [run](results/nvidia/b200) |

Reproduce the table: `python3 bench/per_dollar.py`.

A single H100 finishes a case faster than a p300c board (771.5 s vs 1310 s). The advantage is per dollar: a QuietBox 2
costs 4 % of the H100 server. GPU power is the measured board power while busy, plus a host share. The QuietBox 2 is
charged its full 1.3 kW system rating (not a measurement). NVIDIA gets its cheapest documented price and its
fastest measured precision mode (SPDP: fp32 fields with fp64 solver sums).

**Accuracy.** On this case Cd keeps moving under SIMPLE (standard deviation 42 drag counts over iterations 101-800 in
OpenFOAM's own run; 1 count = 0.0001), so single iterations say little. We compare window means over iterations 101-800 against OpenFOAM v2412 in fp64 on a CPU,
from the same start ([method](docs/accuracy.md)):

| run | Cd mean, its 101-800 | vs OpenFOAM fp64 | script |
|---|---|---|---|
| OpenFOAM v2412 fp64 (CPU) | 0.42116 (+-22 counts, 2 sigma) | reference | |
| tt-fluid, QuietBox 2 board A (fp32) | 0.42013 | -10.3 counts | `bench/nvidia/cdwindow.py` |
| SPUMA SPDP, 8x H100, 8 runs | 0.42079 - 0.42419 | -3.7 to +30.3 counts | `bench/nvidia/grade.py` |
| SPUMA SPDP, B200 | 0.42268 | +15.2 counts | `bench/nvidia/grade.py` |

Over a longer window (iterations 101-1900) the Tenstorrent run's Cd is 0.42060, 12.8 counts below OpenFOAM over the same
iterations. Lift agrees to within 3 counts. Against OpenFOAM's long-run mean (iterations 501-3372) instead, every
window above is still partly transient and the Tenstorrent run is the furthest off, at -21 counts;
[docs/accuracy.md](docs/accuracy.md) has all of it.

## Install

Python 3.10 or newer, Linux x86-64. On a machine with Tenstorrent cards (driver and firmware installed as for any
[tt-metal](https://github.com/tenstorrent/tt-metal) workload):

```bash
git clone https://github.com/moritztng/tt-fluid.git
cd tt-fluid
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu   # CPU torch, ~200 MB
pip install -e '.[tenstorrent]'      # adds ttnn 0.68.0 from PyPI
```

Without a card, `pip install -e .` is enough to run the emulator and the tests.

## Quickstart (5 minutes)

The repository ships OpenFOAM's pitzDaily tutorial (a backward-facing step, 12,225 cells) with its mesh already built.

```bash
ttfluid check cases/pitzDaily -n 3                 # CPU emulator only, no card: ~5 s
ttfluid run cases/pitzDaily -n 300 --chips 1 --write out/pitzDaily
ttfluid run cases/pitzDaily -n 300 --chips 2       # both chips of a p300 board
```

`run` first runs the emulator, then the device, and records both in `ttfluid_runs.jsonl`; the `vs_emu` entry of that
row is the largest relative difference between the two (0 when they agree bitwise). `--write` puts the final U, p, k,
omega and nut into a copy of the case that ParaView or any OpenFOAM tool can open. Wormhole cards take `--arch wh`.
The [getting-started guide](docs/getting-started.md) walks through the output line by line; the
[tutorial](docs/tutorial.md) goes from pitzDaily to the motorbike run behind the table above.

## How it works

```
OpenFOAM case ──> ttfluid.solver  (SimpleFoam in numpy: the reference discretisation, fp32 or fp64)
                     │  trace one SIMPLE iteration
                     v
                  ttfluid.resident (program: a register machine; partition cells over cores and chips;
                     │              lower and encode into three instruction streams per core)
                     ├──> CPU emulator (memu): bitwise reference of the device run
                     v
                  ttfluid.device   (one persistent kernel per core interprets the streams; halos over the NoC
                                    and the chip-to-chip Ethernet fabric; fields return at the end)
```

Large cases are lowered once on the host into per-chip images (`ttfluid.lower`) and then run from those images. Details
in [docs/architecture.md](docs/architecture.md).

## Tests

```bash
pip install -e '.[dev]'
pytest -q          # 33 tests, about a minute, CPU only
```

The tests compare the solver against stored OpenFOAM v2412 results on pitzDaily (mesh geometry to 1e-12, laminar
SIMPLE to 1e-9, wall distance to 1e-5, k-omega SST from OpenFOAM's own first-iteration state to 1e-8) and run the emulator on one and two chips. CI runs them on Python 3.10 and 3.12
with `ruff` and a full-history `gitleaks` scan.

## Limitations

- Steady incompressible flow only: simpleFoam with k-omega SST on the device. The laminar path runs on the CPU solver.
- Boundary conditions: fixedValue, uniformFixedValue, zeroGradient, fixedGradient, inletOutlet, noSlip, slip,
  symmetry, symmetryPlane, empty, calculated, and the kqR, omega, nutk and nutUSpalding wall functions. Cyclic and wedge
  patches are not supported yet.
- The device works in fp32. Results differ from OpenFOAM's fp64 at the level shown in the accuracy table, and a
  one-chip run differs from a two-chip run by about 1 % in residuals after 3 iterations because the partition changes
  the order of fp32 sums. Each partition is deterministic.
- On pitzDaily the first SST iteration agrees with OpenFOAM v2412 to about 1e-2 in k and omega (the next steps,
  started from OpenFOAM's own state, agree to better than 1e-9 and are tested); see [docs/accuracy.md](docs/accuracy.md).
- Lowering a case of several million cells takes host RAM: about 220 GB for HPC_motorbike Small.
- Tested with the ttnn 0.68.0 wheel. This release was verified on a Blackhole p300 board; the same device program ran
  on Blackhole p150 cards and a Wormhole Galaxy during development.

## Roadmap

- A Blackhole Galaxy (32 chips) run of HPC_motorbike Small, and the 65 M-cell occDrivAer car case.
- Lower large cases in parallel across chips (`ttfluid/lower/spmd.py`) so host RAM stops being the limit.
- Close the first-iteration SST gap on pitzDaily.
- More boundary conditions and the simpleFoam variants that keep the same structure (e.g. porosity, MRF).

## Repository layout

| path | what |
|---|---|
| `ttfluid/` | the package: `solver` (numpy SimpleFoam), `resident` (program, partition, lowering, emulator), `lower` (large-case images), `device` (runner and kernels), `cli.py` |
| `cases/` | pitzDaily, ready to run |
| `tests/` | CPU tests against OpenFOAM v2412 references |
| `bench/` | `per_dollar.py`; `nvidia/`: the SPUMA build and run scripts and the Cd graders used for the GPU baseline |
| `results/` | raw logs, per-iteration histories and OpenFOAM references behind every number above |
| `docs/` | [getting started](docs/getting-started.md), [tutorial](docs/tutorial.md), [architecture](docs/architecture.md), [performance](docs/performance.md), [accuracy](docs/accuracy.md), [FAQ](docs/faq.md) |

## Citation

If you use tt-fluid, please cite it with the metadata in [CITATION.cff](CITATION.cff).

## License

GPL-3.0-or-later ([LICENSE](LICENSE)). The pitzDaily case and the case dictionaries come from OpenFOAM (OpenCFD Ltd,
GPL-3.0-or-later). Development history before v0.1.0 lives in a private repository; this one starts from a curated
snapshot.
