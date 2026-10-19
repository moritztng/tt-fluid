# Getting started

## Requirements

- Linux x86-64, Python 3.10 or newer.
- For device runs: a Tenstorrent Blackhole (p150, p300) or Wormhole card with the kernel driver and firmware that
  [tt-metal](https://github.com/tenstorrent/tt-metal) 0.68 expects. `tt-smi` should list the card.
- No OpenFOAM installation. tt-fluid reads and writes OpenFOAM case directories itself. To mesh your own geometry or
  look at results you will want OpenFOAM (blockMesh, snappyHexMesh) and ParaView.

## Install

```bash
git clone https://github.com/moritztng/tt-fluid.git
cd tt-fluid
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu   # with a card: CPU torch first
pip install -e '.[tenstorrent]'      # with a card: adds ttnn 0.68.0
# pip install -e .                   # without a card: emulator and tests only
```

On a QuietBox 2 (fresh venv, packages already in pip's cache) the device install took 31 s.

## First run without hardware

```bash
ttfluid check cases/pitzDaily -n 3
```

`check` lowers the case for one chip and runs it on the CPU emulator. It takes about 5 s and ends like this:

```
lowered + encoded in 1.3 s {... 'arena_kb': 664, ...}
emulated (110 cores) in 4.0 s, 3 rows
...
emu row 0 U 1 its0 1 its1 1 its2 90 its3 3 its4 3 k 0.999999 omega 0.140221 outer 1 p 1
emu row 1 U 0.490278 its0 9 its1 16 its2 93 its3 3 its4 5 k 0.376343 omega 0.0480419 outer 2 p 0.0500561
emu row 2 U 0.266656 its0 9 its1 12 its2 28 its3 2 its4 3 k 0.171532 omega 0.091707 outer 3 p 0.196189
```

How to read it:

- `lowered + encoded`: the SIMPLE iteration was traced and compiled into per-core instruction streams; `arena_kb` is
  the SRAM each core needs.
- `emulated (110 cores)`: the case was split over 110 cores of one Blackhole chip, about 112 cells each.
- `emu phase ...` lines count the work per phase of the iteration (momentum, U solve, pressure, turbulence).
- each `emu row` is one SIMPLE iteration: the initial residuals of U, p, k and omega, normalised the way OpenFOAM
  normalises them, and `its0`...`its4`, the linear-solver iterations of each solve in the order they run (the velocity
  components, p, omega, k).

## First run on a card

```bash
ttfluid run cases/pitzDaily -n 300 --chips 1 --write out/pitzDaily
```

This runs the emulator (about 3.5 minutes on one CPU core), then the device (3.5 s on a p300 chip at 1350 MHz,
compile included), then writes the final fields into `out/pitzDaily/<N>/`, N being the number of iterations
run, inside a copy of the case. Open `out/pitzDaily` in ParaView, or run any OpenFOAM utility on it.

Each run appends one JSON row to `ttfluid_runs.jsonl` (`--out` to change). Useful keys:

| key | meaning |
|---|---|
| `last` | residuals and solver iterations of the final iteration |
| `rows` | number of SIMPLE iterations run (residualControl in `system/fvSolution` can stop the run before `-n`) |
| `abort` | `cores`: how many cores gave up on a bounded wait (0 on a good run); a run with aborts exits with status 3 |
| `written` | where `--write` put the fields |
| `walls` | wall time of the device run, seconds |
| `aiclk` | the chip clock read during the run, per chip (min, max) |
| `vs_emu` | device against emulator: `worst_rel` over the residual history and `state` per field (max-norm, relative); small, not 0 ([accuracy.md](accuracy.md)) |

Other options:

- `--chips 2` uses both chips of a p300 board (a 1x2 mesh). `--chips 4`, `8`, `16`, `32` map to 2x2, 2x4, 4x4 and 4x8
  meshes for multi-card systems and Galaxies.
- `--arch wh` for Wormhole.
- `--no-check` skips the emulator (for large cases the emulator is slow).
- Everything after `--` goes to the underlying runner, `python -m ttfluid.device.run_mesh --help`.

By default `run` uses cards `0..chips-1`. Set `TT_VISIBLE_DEVICES` to choose others.

## Your own case

Any steady simpleFoam case with k-omega SST and supported boundary conditions (see the README's limitations) should
run as it is: `constant/polyMesh` from blockMesh or snappyHexMesh, `0/` with U, p, k, omega, nut, and the usual
`system/` dictionaries. The schemes and solver tolerances come from `fvSchemes` and `fvSolution`. The SIMPLE loop,
relaxation factors and residualControl run as in simpleFoam.

The defaults keep every vector in the cores' SRAM. pitzDaily needs 664 KB of each Blackhole core's 1.5 MB at 112 cells
per core, so one chip holds a few tens of thousands of cells this way. Larger cases need the DRAM-streamed lowering; see
the [tutorial](tutorial.md).

## Troubleshooting

- **`ModuleNotFoundError: ttnn`** or **`torch`**: install with `pip install -e '.[tenstorrent]'`. Install the CPU
  build of torch first (as above), or pip fetches the much larger CUDA build.
- **The run exits with status 3 and `abort.cores` > 0**: some cores gave up on a bounded wait, and the program ended
  without hanging the chip. Reset the card (`tt-smi -r`) and rerun. If it repeats, open an issue with the log.
- **Unsupported boundary condition**: the run stops while reading the case. Open an issue with the patch type; most
  simple types are a few lines in `ttfluid/solver/fields.py`.
