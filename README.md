# tt-fluid

[![CI](https://github.com/moritztng/tt-fluid/actions/workflows/ci.yml/badge.svg)](https://github.com/moritztng/tt-fluid/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

**tt-fluid runs OpenFOAM's steady incompressible solver (simpleFoam, k-omega SST) entirely on Tenstorrent chips.** It
reads an ordinary OpenFOAM case directory and runs every SIMPLE iteration on the device: momentum assembly, the
turbulence model, all four linear solves and the flux correction. It writes the fields back as OpenFOAM time directories.
OpenFOAM is not needed at run time. The same program also runs on a CPU emulator, which needs no hardware and follows the
device closely but not bit for bit.

On the OpenFOAM HPC benchmark's motorbike case (HPC_motorbike Small: 8.66 M cells, 800 SIMPLE iterations), a $9,999 Tenstorrent QuietBox 2
costs **$0.115 per case**. An 8x H100 SXM server running the GPU port of OpenFOAM (SPUMA) at its cheapest documented
price costs $0.270 per case with one case per GPU, $0.119 with four cases per GPU under NVIDIA MPS and **$0.096 with
eight: a GPU packed with cases is cheaper per case than the QuietBox 2 today, by 1.2x.** The drag coefficient lands 10 drag counts (0.2 %) below OpenFOAM's
fp64 CPU result, inside the reference's own uncertainty, while the eight H100 runs scatter from -4 to +30.

> **Status, 2026-11-28.** The packed-GPU numbers are new. They were measured on one H100 SXM, priced as one eighth of
> the server: two cases per GPU $0.158 per case, four $0.119, eight $0.096, every case passing the drag check. A whole
> 8-GPU server with eight cases per GPU (64 busy host cores) is still to be measured. These numbers are not yet in the
> table, the figure and the scripts below, which still compare against one case per GPU; their raw data and the
> updated table follow. Every other number in this README is printed by a script in this repository from raw data in
> this repository.

![Cost per case and drag coefficient vs OpenFOAM](docs/figures/headline.svg)

*The figure shows one case per GPU on the NVIDIA side (see the status note above).*

## Why it matters

Steady RANS is still most of the CFD that engineering teams run, and GPU ports of OpenFOAM are bound by memory
bandwidth. A Tenstorrent chip is a grid of more than a hundred small cores, each with its own SRAM, linked by a network on chip,
and chips link to each other over Ethernet. tt-fluid keeps the whole iteration resident on that grid: each core owns a
block of cells, exchanges halos with its neighbours directly, and never returns to the host inside a run. The result is
a $9,999 box doing the same job within 1.2x of the cost per case of a fully packed GPU server, with its accuracy
checked against OpenFOAM.

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

A single H100 finishes a case faster than a p300c board (771.5 s vs 1310 s). With one case per GPU the QuietBox 2 wins
per dollar because it costs 4 % of the H100 server. Packing the GPUs reverses that: with four cases per GPU the
QuietBox 2 is 1.03x cheaper per case, with eight it is 1.2x more expensive (status note above). GPU power is the measured board power while busy, plus a host share. The QuietBox 2 is
charged its full 1.3 kW system rating (not a measurement). NVIDIA gets its cheapest documented price and its
fastest measured precision mode (SPDP: fp32 fields with fp64 solver sums).

**Accuracy.** On this case Cd keeps moving under SIMPLE (standard deviation 42 drag counts from iteration to
iteration in OpenFOAM's own run; 1 count = 0.0001), so runs are compared on window means. `bench/cdgrade.py` grades the
mean from iteration 201 on against the long-run mean of OpenFOAM v2412 in fp64 on a CPU (Cd 0.42226), from the same
start. One run is a screen with a band of 17.8 counts. The bar is 12.5 counts on four or more runs pooled
([method](docs/accuracy.md#3-drag-and-lift-against-openfoam-fp64-hpc_motorbike-small)):

| runs | Cd vs OpenFOAM fp64, counts | grade |
|---|---|---|
| tt-fluid fp32, 3 runs (QuietBox 2, Blackhole Galaxy on 2 and on 4 chips) | -17.0, +3.8, -4.3; pooled -6.0 +- 8.4 | pass, band 16.9 (three runs: a screen) |
| SPUMA SPDP, 9 runs (8 on H100, 1 on B200) | -12.6 to +27.0; pooled +8.4 +- 5.0 | pass, band 12.5 |

**occDrivAer.** The coarse occDrivAer car (1.6 M cells, [cases/occDrivAer](cases/occDrivAer)) ran iterations 1-3999 on
one Blackhole chip. Its mean Cd over iterations 1001-3999 is 0.2738, 15.0 counts above a serial OpenFOAM v2412 fp64
run and 0.5 counts from an 8-rank OpenFOAM run restarted from that run's iteration 1000. The two OpenFOAM runs differ
from each other by 15.1 counts, so on this case the device is as close to OpenFOAM as OpenFOAM is to itself
([details](docs/accuracy.md#4-drag-and-lift-against-openfoam-occdrivaer-coarse-mesh)).

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

## Quickstart (about 10 minutes)

The repository ships OpenFOAM's pitzDaily tutorial (a backward-facing step, 12,225 cells) with its mesh already built
and the turbulence model set to k-omega SST.

```bash
ttfluid check cases/pitzDaily -n 3                 # CPU emulator only, no card: ~5 s
ttfluid run cases/pitzDaily -n 300 --chips 1 --write out/pitzDaily    # ~4 min
ttfluid run cases/pitzDaily -n 300 --chips 2       # both chips of a p300 board, ~7 min
```

Almost all of that time is the CPU emulator, which `run` executes first as a reference: on a p300 the device part of
the one-chip run takes 3.5 s, compile included. Add `--no-check` to skip the emulator. Run to convergence (`-n 1000`; residualControl stops it at iteration 355 on one
chip), the device's fields lie within 0.4 % (U) and 0.8 % (p) of OpenFOAM v2412's converged solution
([accuracy.md](docs/accuracy.md#2a-converged-pitzdaily-against-openfoam)).

`run` first runs the emulator, then the device, and records both in `ttfluid_runs.jsonl`; the `vs_emu` entry of that
row is the largest relative difference between the two residual histories. It is small but not 0: the chip's vector unit
computes division, square roots and exp/log/tanh with its own approximations, the emulator with numpy ([accuracy.md](docs/accuracy.md)). `--write` puts the final U, p, k,
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
                     ├──> CPU emulator (memu): reference of the device run, to fp32 rounding
                     v
                  ttfluid.device   (one persistent kernel per core interprets the streams; halos over the NoC
                                    and the chip-to-chip Ethernet fabric; fields return at the end)
```

Large cases are lowered once on the host into per-chip images (`ttfluid.lower`) and then run from those images. Details
in [docs/architecture.md](docs/architecture.md).

## Tests

```bash
pip install -e '.[dev]'
pytest -q          # 34 tests, about a minute, CPU only
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

- A Blackhole Galaxy (32 chips) run of HPC_motorbike Small, and the larger occDrivAer meshes (up to 65 M cells).
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
