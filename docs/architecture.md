# Architecture

tt-fluid has three layers. A numpy solver defines the discretisation, a register machine turns one SIMPLE iteration of
that solver into a program for every core, and a persistent device kernel runs the program. The CPU emulator of that
program is the bridge between the two worlds: it is checked against the numpy solver, and every device run is compared
with it.

```
OpenFOAM case
   │  ttfluid.solver.foamio      read polyMesh, 0/, constant/, system/ (fvSchemes, fvSolution, controlDict)
   v
ttfluid.solver.SimpleFoam        SIMPLE/SIMPLEC + k-omega SST, numpy, fp64 or fp32
   │  ttfluid.resident.program   trace ONE outer iteration into a register-machine IR
   v
ttfluid.resident.machine         partition cells over chips and cores (RCB), schedule, lower to per-core MIR
   │  ttfluid.resident.encode    three uint32 instruction streams per core + constant tables
   ├─> ttfluid.resident.memu     CPU emulator: the exact reference of the device run
   v
ttfluid.device.run_mesh          ttnn generic_op: kernels/vm_{brisc,ncrisc,compute}.cpp, one persistent program per core
   │                             loops over SIMPLE iterations on the device, stops on residualControl or the cap
   v
fields back to the host  ──>  OpenFOAM time directory (--write), per-iteration residual history (jsonl)
```

## The solver: one algorithm, two operator backends

`ttfluid/solver/simple.py` (SIMPLEC) and `ttfluid/solver/turbulence.py` (k-omega SST) are written once against the
operator interface in `ttfluid/solver/ops.py`: interpolate, flux, grad, snGrad, surface integral, and the implicit
`div`, `laplacian`, `Sp`, `SuSp`, which return a matrix with `relax`, `A`, `H`, `H1`, `flux` and `solve`.

`FaceOps` implements the interface face by face, the way OpenFOAM's `fvMatrix` and `fvm`/`fvc` operators do
(`fvm.py`, `fvc.py`). It is the oracle, and the tests check it against OpenFOAM v2412.

`EllOps` implements the same interface in the layout the device runs (`ell.py`), in fp64 or fp32. Every operation is
one of four primitives:

| primitive | what it does | on the device |
|---|---|---|
| gather | `x_ext[nbr]`: the value on the other side of each face of each cell | data-movement core, from its own SRAM or a halo buffer |
| per-slot arithmetic | interpolation, fluxes, coefficient assembly, upwinding, limiters | SFPU, fp32 |
| row sum | sum over a cell's faces | SFPU |
| dot / norm | Krylov reductions, residual norms | SFPU plus a cross-core and cross-chip reduction |

### The cell-centric ELL layout

Each cell row holds one slot per face. A slot stores the index of the value across the face, the face's area vector
oriented outward from this cell, the interpolation weight, the non-orthogonal delta coefficient and the correction
vector. Interior faces appear in both adjacent rows. Boundary faces point at ghost entries that hold the patch face
values, so `x_ext = [cell values ; boundary face values]`, and a boundary condition only ever writes ghost entries and
per-slot coefficients.

- No scatter. A face loop adds each face result into two cells, which on a many-core machine means atomics or
  colouring. Here each row sums only its own slots.
- One irregular access pattern for the whole solver. Gradients, convection and diffusion assembly, the SpMV of every
  Krylov iteration and the face fluxes all gather through the same `nbr` table, so one gather path and one halo
  exchange serve everything.
- Conservation is exact. Both copies of an interior face flux use the same operands in the same order and differ only
  in the sign of the area vector, so the sum over cells telescopes as it does in OpenFOAM.
- The redundant arithmetic on interior faces is cheap SFPU work. What costs time is moving bytes, and the gather is
  needed either way.

## The resident machine

`ttfluid.resident.program.build` runs one outer iteration of `SimpleFoam` with `EllOps` on a tracer instead of numpy
arrays and records every vector operation as an instruction on named registers (cell vectors, face-slot vectors,
scalars). Linear solves become loops with device-side convergence tests, so the number of Krylov iterations is decided
on the chip exactly as OpenFOAM's `tolerance` and `relTol` would decide it.

`ttfluid.resident.machine.Machine` then:

1. partitions the cells over chips and cores with recursive coordinate bisection (`--layout rcb`), about 110 cores per
   Blackhole chip;
2. turns every gather into local reads plus halo receives, and every reduction into a per-core partial sum followed by
   a tree over cores and chips;
3. schedules the instructions (`--lsched`, `--phase`), fuses chains of element-wise work (`--fuse`, `--cse`), and
   allocates SRAM. Small cases keep every vector in SRAM. Large cases home vectors in DRAM and stream them through SRAM
   pages (`--tabstream`, `--page`, `--nwin`).

`ttfluid.resident.encode` packs the result into three uint32 streams per core, one for each kernel:

- `vm_brisc.cpp` (BRISC) loads the core's arena and code, then runs the B stream: signals and waits, halo pushes with
  receiver credits, the all-to-all reduction of per-core partial sums, DRAM constant fetches and the residual history.
- `vm_ncrisc.cpp` (NCRISC) runs the N stream: gathers from the per-core map tables (own cells, halo slots, zeros),
  halo credits back to the senders, and the pages that stream DRAM-homed vectors into SRAM.
- `vm_compute.cpp` (the three TRISCs) decode the C stream together: unpack, SFPU arithmetic in fp32, pack.

The kernels are interpreters. A case changes only a few compile-time constants, and ttnn caches the build. Every wait
in them is bounded: a core that gives up records where and why, raises an abort flag that every other core checks, and
the program still reaches its end, so a bad program returns a diagnosis instead of a hung chip.

`ttfluid.resident.memu.MachineEmu` interprets the same streams on the CPU with the same fp32 rounding, in the same
order. A device run without `--no-check` runs the emulator first and records the largest difference (`vs_emu` in the
result row). The test suite runs the emulator on pitzDaily on one and two chips.

## Chips, boards and meshes

`--chips K` maps to a ttnn mesh: 1 = `1,1`, 2 = `1,2` (a p300 board), 4 = `2,2`, 8 = `2,4`, 32 = `4,8` (a Galaxy).
Halos and reductions between chips use the Ethernet fabric (`FABRIC_1D`) through gateway cores (`--gw`). On a p300
board `ttfluid run` picks the matching mesh graph descriptor shipped with ttnn (`p150_...` for one chip, `p300_...`
for both); set `TT_MESH_GRAPH_DESC_PATH` yourself to override.

## Large cases

A case of millions of cells is lowered once on the host and written as per-chip images (`ttfluid/lower/image.py`), so
the device run never holds the whole program in memory and reads only its chips' bytes:

1. `run_mesh case N ... --images DIR` lowers and writes the images, then exits. For HPC_motorbike Small this takes about
   220 GB of RAM.
2. `lower/of_state.py` turns an OpenFOAM time directory (for example the potentialFoam result) into a starting state.
3. `run_mesh case N ... --from-images DIR --state-in STATE.npz` runs on the device.

`lower/prep.py` and `lower/spmd.py` split the lowering into one process per chip for cases too large for one host
process. That path is experimental in v0.1.0.

## Precision

The device works in fp32 on the SFPU. The matrix engine reads operands at reduced precision and is not used for solver
arithmetic. Reductions are summed per core and then over a fixed tree, so a run is deterministic for a given partition.
A different partition changes the order of fp32 sums and therefore the low bits of the result. See
[accuracy.md](accuracy.md).
