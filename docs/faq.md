# FAQ

**Is this OpenFOAM?**
No. tt-fluid is an independent implementation of simpleFoam's algorithm and discretisation, written in Python (numpy)
for the reference and compiled to Tenstorrent kernels for the run. It reads and writes OpenFOAM case directories and is
tested against OpenFOAM v2412 results, so OpenFOAM stays your mesher and post-processor. OpenFOAM is a trademark of
OpenCFD Ltd; this project is not affiliated with or endorsed by OpenCFD.

**Which solvers and models are supported?**
Steady incompressible SIMPLE/SIMPLEC (simpleFoam) with k-omega SST on the device. Laminar runs on the CPU solver only.
See the README's limitations for boundary conditions.

**Do I need a Tenstorrent card to try it?**
No. `pip install -e .` and `ttfluid check` run the exact device program on a CPU emulator. It is slow (about a second
per pitzDaily iteration) but gives the same numbers, bit for bit, that the chip produces.

**Why fp32? OpenFOAM runs in fp64.**
The chip's vector unit works in fp32. On the benchmark case the drag coefficient is 10 counts (0.2 %) below OpenFOAM's fp64
result, inside the reference's own uncertainty; GPU runs in OpenFOAM's mixed precision scatter from -4 to +30 counts
([accuracy.md](accuracy.md)). Mixed-precision OpenFOAM (SPDP) on GPUs makes the same
trade.

**Why is a single H100 faster per case but the QuietBox 2 cheaper per case?**
Price. One H100 SXM finishes the 8.66 M-cell case in about 750 s, one p300c board in about 1300 s, but a QuietBox 2 with
two boards costs $9,999 and an 8-GPU H100 server $249,200. The full calculation is in [performance.md](performance.md).

**How do I know the numbers are real?**
Every number in the README comes out of a script in `bench/` run on files in `results/`: the raw device logs with
clocks, the GPU runs with their per-second power and clock samples, and the OpenFOAM reference. Run them and check.

**Can I run on a Wormhole n150/n300 or a Galaxy?**
Use `--arch wh`. `--chips` 4 to 32 map to the 2x2 to 4x8 meshes of multi-card systems and Galaxies. This release was
verified on a Blackhole p300. The same device program ran on Wormhole Galaxy and Blackhole p150 systems during
development, but those runs are not part of this release's results.

**How big a case can I run?**
Up to a few tens of thousands of cells per chip stay entirely in SRAM with the default flags. Larger cases stream
vectors from DRAM (`--tabstream`, see the [tutorial](tutorial.md)). The 8.66 M-cell motorbike runs on two chips. The
practical limit today is host memory for the lowering (about 220 GB for that case).

**Why does a 2-chip run give slightly different residuals than a 1-chip run?**
fp32 addition is not associative, and a different partition sums in a different order. Each partition gives the same
answer every time.

**What happened to the development history?**
It lives in a private repository. This repository starts from a curated snapshot at v0.1.0.

**How do I cite it?**
See [CITATION.cff](../CITATION.cff).
