# Tutorial: from pitzDaily to HPC_motorbike

Part 1 runs OpenFOAM's pitzDaily tutorial to convergence on one chip and compares it with the emulator. Part 2 is the
recipe behind the cost table in the README: an 8.66 M-cell motorbike on a QuietBox 2.

## Part 1: pitzDaily

pitzDaily is a backward-facing step: air enters at 10 m/s on the left, separates at the step and reattaches
downstream. The shipped case is OpenFOAM v2412's tutorial with the turbulence model set to kOmegaSST (the tutorial selects
kEpsilon), and with the 12,225-cell blockMesh mesh included.

### Run to convergence

`system/fvSolution` asks simpleFoam to stop when the initial residuals fall below `residualControl` (p 1e-2, U, k and
omega 1e-3). The device evaluates the same test after every iteration, so give it an upper bound and let it stop:

```bash
ttfluid run cases/pitzDaily -n 1000 --chips 1 --write out/pitzDaily
```

The emulator runs first (it is the slow part here: just under a second per iteration on one CPU core), then the device. The last
line of the output is the result row, also appended to `ttfluid_runs.jsonl`. `rows` is the number of iterations run before residualControl stopped it.
`vs_emu` compares the device with the emulator; it is small but not 0 (see [accuracy.md](accuracy.md)).

On one p300 chip the device stopped at iteration 355 (2 chips: 400) and its fields lie within 0.4 % (U) and 0.8 % (p) of OpenFOAM v2412's converged solution ([accuracy.md](accuracy.md#2a-converged-pitzdaily-against-openfoam)).

Without a card, run the same thing on the emulator alone:

```bash
ttfluid check cases/pitzDaily -n 1000
```

On our machine the emulator took 323 s and ended at iteration 371 with initial residuals U 9.9e-4, p 1.7e-3, k 1.5e-4,
omega 3.6e-4.

### Look at the result

`out/pitzDaily` is a normal OpenFOAM case with the converged fields in `out/pitzDaily/<N>/`. With OpenFOAM installed:

```bash
cd out/pitzDaily
touch pitzDaily.foam && paraview pitzDaily.foam     # or: foamToVTK -latestTime
postProcess -func 'components(U)' -latestTime         # any OpenFOAM utility works on it
```

Running OpenFOAM's own `simpleFoam` on `cases/pitzDaily` gives the reference to compare against. Expect differences at
the level of fp32 arithmetic and solver tolerances: OpenFOAM's own GAMG and PCG runs of this case differ from each other
by about 0.7 % in U at convergence.

### One chip or two

```bash
ttfluid run cases/pitzDaily -n 1000 --chips 2
```

On a p300 board this splits the cells over both chips and exchanges halos over the board's Ethernet link. The answer
differs from the one-chip run in the low bits, because fp32 sums are taken in a different order (see
[accuracy.md](accuracy.md)).

## Part 2: HPC_motorbike Small on a QuietBox 2

This is the run behind the README's cost table. It needs a host with about 220 GB of RAM for the one-time lowering and a
QuietBox 2 (or any machine with p300 boards) for the run. Each step below is the step that produced the published
numbers, and the logs in `results/tenstorrent/quietbox2-p300x4/` hold the exact flags of those runs (the `jsonl` rows
record every option). Unlike everything else in these docs, Part 2 was not re-run from a clean clone for v0.1.0: it
needs the big-memory host and several hours. If a step does not work as written, please open an issue.

### 1. Get the case

The case is `incompressible/simpleFoam/HPC_motorbike/Small` of the OpenFOAM HPC Technical Committee
([develop.openfoam.com/committees/hpc](https://develop.openfoam.com/committees/hpc)). Mesh it with its own scripts
(blockMesh, snappyHexMesh) in OpenFOAM v2412, then initialise with `potentialFoam -writephi`. You now have
`constant/polyMesh`, `0/` and the potentialFoam fields.

### 2. Lower once, write per-chip images

```bash
FLAGS="--arch bh --mesh 1,2 --layout rcb --l1 0 --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 1024 --tabstream \
  --amg --amg-stack --amg-vec --amg-smoothed 0 --amg-min-core 10 --amg-depth 0 --vcoarse 8 \
  --lsched --phase --sliced --fuse --cse 1 --hlate 1 --vpre 1 --vpost 1 --strips"
python -m ttfluid.device.run_mesh case 800 --case motorbike $FLAGS --images img_k2
```

This traces one SIMPLE iteration of the full case, partitions it over the two chips of one board, schedules and encodes
it, and writes one image per chip. It runs on the CPU only. It refuses to start if the write would leave the disk with
less than 200 GB free (`--min-free-gb`).

The pressure solve is Jacobi-preconditioned CG (`--amg-depth 0` builds no coarse level). On Blackhole that is faster
than an aggregation V-cycle: it needs about six times as many iterations, but each costs a twentieth.

### 3. Starting state from potentialFoam

```bash
python -m ttfluid.lower.of_state img_k2 motorbike motorbike/0 state0.npz
```

### 4. Run on the device

```bash
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$(python -c "import ttnn, pathlib; print(pathlib.Path(ttnn.__file__).parent / \
  'tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto')")
python -m ttfluid.device.run_mesh case 800 $FLAGS --from-images img_k2 --state-in state0.npz --noemu --runs 1 \
  --links 0 --rlink 1 --os 1 --tsort 1 --gcache 14 --bgts 5 --tgts 2 --l1 1 \
  --forces motorbike/system/controlDict:forceCoeffsAll --out motorbike.jsonl
```

On one p300c board at 1350 MHz this takes about 1250 s of device time for 800 iterations. A QuietBox 2 runs two such
cases at once, one per board (`TT_VISIBLE_DEVICES=2,3` for the second).

`--forces` evaluates the case's `forceCoeffs` function object on the state each iteration starts from and stores Cd
and Cl in the history, which is saved with the final state in `motorbike_case_1x2_state.npz` (key `hist`). Extract it
for the grader:

```bash
python -c "import numpy as np; open('motorbike.hist.json', 'w').write(str(np.load('motorbike_case_1x2_state.npz')['hist']))"
```

### 5. Grade and price it

```bash
python3 bench/nvidia/cdwindow.py --dev motorbike.hist.json:1 --of coefficient.dat --from 101 --to 800
python3 bench/per_dollar.py
```

`coefficient.dat` is OpenFOAM's `postProcessing/forceCoeffsAll/0/coefficient.dat` from a CPU run of the same case,
from the same potentialFoam start. [accuracy.md](accuracy.md) explains the windowed comparison, and
[performance.md](performance.md) the cost model.
