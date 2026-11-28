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

This is the largest case in the README's cost table. It needs a host with about 90 GB of RAM for the one-time lowering
(it peaks at 84 GB) and a QuietBox 2, or any machine with p300 boards, for the run. The steps below are the fastest
line measured so far. The README's table was made with the v0.1.0 line, and the logs in
`results/tenstorrent/quietbox2-p300x4/` hold the exact flags of those runs (the `jsonl` rows record every option).
Part 2 was not re-run from a clean clone: it needs the big-memory host and about an hour. If a step does not work as
written, please open an issue.

### 1. Get the case

The case is `incompressible/simpleFoam/HPC_motorbike/Small` of the OpenFOAM HPC Technical Committee
([develop.openfoam.com/committees/hpc](https://develop.openfoam.com/committees/hpc)). Mesh it with its own scripts
(blockMesh, snappyHexMesh) in OpenFOAM v2412, then initialise with `potentialFoam -writephi`. You now have
`constant/polyMesh`, `0/` and the potentialFoam fields.

### 2. Lower once, write per-chip images

```bash
FLAGS="--arch bh --mesh 1,2 --layout rcb --wplace 2 --gw 1 --nsh 1 --page 16 --tabstream --lsched --phase --sliced \
  --fuse --cse 2 --hlate 1 --strips --tsort 1 --l1 1 --l1-bytes 1461760 --lsplit 2 --tabdense 530 --span 2 \
  --forces motorbike/system/controlDict:forceCoeffsAll"
PYTHONHASHSEED=0 python -m ttfluid.device.run_mesh case 800 --case motorbike $FLAGS --nwin 8192 --images img_k2
```

This traces one SIMPLE iteration of the full case, partitions it over the two chips of one board, schedules and encodes
it, and writes one image per chip. It runs on the CPU only and takes about half an hour. It refuses to start if the
write would leave the disk with less than 200 GB free (`--min-free-gb`). `PYTHONHASHSEED=0` fixes the order of the
reductions, so two lowerings give the same images.

The pressure solve is Jacobi-preconditioned CG, the default when `--amg` is not given. On Blackhole that is faster
than an aggregation V-cycle: it needs about six times as many iterations, but each costs a twentieth.

Four of these options are what makes this line faster than v0.1.0's, and none of them changes a value.
`--tabdense 530` stores the larger gather maps as dense index tables. `--span 2` gives a gather from a vector that
lives in DRAM a copy in L1 memory to read. `--lsplit 2` moves a vector out of L1 for the duration of a solver loop
that needs the room for its own vectors. `--cse 2` computes an interpolation or a gradient of a field once when
several terms ask for it. With the settings the run chooses itself (step 4), the first 100 iterations take 108 s on
the device. The line this one replaces (`--l1 2 --lsplit 1`) took 136 s and v0.1.0's (`--l1 1 --gcache 14`) 161 s,
so this one is 1.26 and 1.49 times faster, and the state after those 100 iterations is bit for bit the same with all
three. [architecture.md](architecture.md) describes the options, and the raw logs of these legs are in
`results/tenstorrent/galaxy-blackhole-pair/`.

### 3. Starting state from potentialFoam

```bash
python -m ttfluid.lower.of_state img_k2 motorbike motorbike/0 state0.npz
```

### 4. Run on the device

```bash
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$(python -c "import ttnn, pathlib; print(pathlib.Path(ttnn.__file__).parent / \
  'tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto')")
python -m ttfluid.device.run_mesh case 800 --case motorbike $FLAGS --from-images img_k2 --state-in state0.npz \
  --noemu --runs 1 --nwin 1024 --links 0 --rlink 1 --os 1 --out motorbike.jsonl
```

On two Blackhole chips at 1350 MHz the first 100 iterations take about 108 s of device time (measured on two
chips of a Blackhole Galaxy). 800 iterations have not been timed with this line yet; with the line it replaces they
took 1030 s, at 136 s for the first 100. A QuietBox 2 runs two such cases at once, one per board
(`TT_VISIBLE_DEVICES=2,3` for the second). The run sizes its gather cache to the L1 memory that is left, chooses
`--bpatch` from counts stored in the images and shares the copies of a gather among the three processors of a core
(`--bgts 6 --tgts 3` on Blackhole). It prints each choice.

`--forces` evaluates the case's `forceCoeffs` function object on the state each iteration starts from and stores Cd
and Cl in the history, which is saved with the final state in `motorbike_case_1x2_state.npz`.

### 5. Grade and price it

```bash
python3 bench/cdgrade.py --run motorbike_case_1x2_state.npz:0
python3 bench/per_dollar.py
```

`cdgrade.py` compares the mean drag over iterations 201-799 with OpenFOAM v2412 in fp64 (`results/openfoam/`, a CPU run
of the same case from the same potentialFoam start) and exits 0 on a pass. [accuracy.md](accuracy.md) explains the
windowed comparison, and [performance.md](performance.md) the cost model.
