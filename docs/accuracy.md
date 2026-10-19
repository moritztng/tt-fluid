# Accuracy and precision

Three questions, three kinds of check:

1. Does the solver discretise like OpenFOAM? The numpy solver is checked against OpenFOAM v2412 field by field.
2. Does the device run the program it was given? The device is compared with the CPU emulator of the same program.
3. Is the fp32 answer good enough? Drag and lift of a motorbike are compared with OpenFOAM in fp64 over windows of iterations.

## 1. The solver against OpenFOAM v2412 (pitzDaily)

The references in `tests/data/pitzDaily_v2412.npz` come from OpenFOAM v2412 runs of the stock pitzDaily tutorial with
every linear solve converged to 1e-13, so the comparison measures the discretisation and not solver tolerances.

| check | test | tolerance | measured |
|---|---|---|---|
| cell centres and volumes | `test_vs_openfoam.py::test_mesh` | 1e-12 | about 1e-16 |
| laminar SIMPLEC, face backend, 3 iterations | `test_laminar[face]` | 1e-9 | 1.4e-11 |
| laminar SIMPLEC, device-layout backend, 3 iterations | `test_laminar[ell]` | 1e-9 | about 1e-11 |
| wall distance (meshWave) | `test_wall_distance` | 1e-5 | 4.9e-6 |
| k-omega SST iterations 2-3 from OpenFOAM's iteration-1 fields, both backends | `test_sst_from_openfoam_state` | 1e-8 | 5.8e-10 (nut), 5.4e-11 (k) |

With k-omega SST, a step started from OpenFOAM's own state agrees to about 1e-9 to 1e-11 (k, omega, nut); the test
starts from OpenFOAM's fields after iteration 1 (`tests/data/sst_t1`) and checks iterations 2 and 3.
The first step from the initial conditions does not agree yet: U and p match to 1e-13, but k differs by up to 2.6e-3
and nut by up to 8.9e-3 in about 150 of the 12,225 cells. Those cells form a few contiguous runs about 13 mm above the
lower wall, near where OpenFOAM bounds a negative omega on that step ("bounding omega, min: -3614"). The difference is
local to the first step and is not yet explained, which is why the SST test starts after it.

## 2. The device against the emulator

`ttfluid.resident.memu` interprets the same instruction streams as the device kernels, with the same fp32 operations
in the same order and the same fixed order for folding per-core partial sums. It is not bitwise: the chip's vector unit
computes division, square roots, exp, log and tanh with its own approximations (`ttfluid/device/kernels/vm_compute.cpp`),
the emulator with numpy. `ttfluid run` runs the emulator first and stores the largest relative difference to the device
in the result row (`vs_emu.worst_rel` over the residual history, `vs_emu.state` per field, max-norm).

Measured for this release on one Blackhole p300 chip (AICLK 1350 MHz, stock `ttnn` 0.68.0 wheel, pitzDaily, k-omega
SST): the first iteration agrees in U, p and k and in every solver iteration count, and the omega residual differs by
4e-5. SIMPLE carries that forward: at iteration 2 the p residual differs by 5 % and one solve takes two more
iterations, and after 300 iterations, still in the transient, the fields differ by up to 27 % (U, max-norm). The emulator is therefore the reference for what the program computes, and the
answer is graded against OpenFOAM on the device's own output (section 3). The emulator itself is regression-tested in
`tests/test_emulator.py` on one and two chips.

The two partitions give different numbers: after 3 pitzDaily iterations the 1-chip and 2-chip residuals differ by about
1 %. That is fp32 summation order, not an error. Each partition is deterministic, run after run.

## 3. Drag and lift against OpenFOAM fp64 (HPC_motorbike Small)

HPC_motorbike Small (8.66 M cells, k-omega SST) does not settle to a steady state under SIMPLE: over iterations 101-800
OpenFOAM's own drag coefficient has a standard deviation of 42 drag counts (1 count = 0.0001 in Cd). A comparison of
single iterations would mean nothing. Both sides are therefore averaged over a window of iterations, and the error of each mean
comes from batch means (blocks of 100 iterations). The OpenFOAM reference is v2412 in fp64 on a CPU from the same
potentialFoam start (`results/openfoam/of1k.dat`, `of4k.dat`).

Iterations 101-800, all runs from the same start:

| run | Cd | vs OpenFOAM, counts | 2 sigma of the difference |
|---|---|---|---|
| OpenFOAM v2412 fp64 | 0.42116 | | 22.0 (OpenFOAM's own mean) |
| tt-fluid, QuietBox 2 board A, fp32 | 0.42013 | -10.3 | 23.8 |
| SPUMA SPDP on H100, 8 runs of the same case | 0.42079 to 0.42419 | -3.7 to +30.3 | 24.2 to 29.6 |
| SPUMA SPDP on B200 | 0.42268 | +15.2 | 26.7 |

Over iterations 101-1900 the Tenstorrent run gives Cd 0.42060, 12.8 counts below OpenFOAM over the same iterations.
Lift (Cl) agrees within 3 counts in both windows.

Against OpenFOAM's long-run mean instead (its iterations 501-3372, Cd 0.42226), iterations 101-800 are still partly
transient for every run: tt-fluid is -21.3 counts (2 sigma 11.6), the eight H100 runs -14.7 to +19.3, the B200 +4.2.
On that comparison the Tenstorrent run is the furthest from OpenFOAM. The last line `cdwindow.py` prints applies a
3-count target to it, which this run does not meet.

The eight H100 runs are the same case on identical GPUs, and their window means still spread over 34 counts: that is
how much a window mean moves with nothing but the order of floating-point sums. The Tenstorrent result (-10.3) lies
just outside that spread on the low side and inside OpenFOAM's own +-22.

Reproduce:

```bash
Q=results/tenstorrent/quietbox2-p300x4; O=results/openfoam
python3 bench/nvidia/cdwindow.py --dev $Q/qb2k2_D1A.hist.json:1 --of $O/of1k.dat $O/of4k.dat --from 101 --to 800
python3 bench/nvidia/cdwindow.py --dev $Q/qb2k2_D1A.hist.json:1 $Q/qb2k2_D2A.hist.json:801 --of $O/of1k.dat $O/of4k.dat --from 101 --to 1900
for g in 0 1 2 3 4 5 6 7; do python3 bench/nvidia/grade.py --of $O/of1k.dat $O/of4k.dat --to 800 \
    --run results/nvidia/dgx-h100/A_SPDP_g$g/coefficient.dat; done
```

`cdwindow.py` prints one-sigma errors; `grade.py` prints two-sigma errors. The Tenstorrent history files hold Cd and Cl
of the state each iteration starts from (`Cd_in`), which is OpenFOAM's row for the same iteration number.

The QuietBox 2 runs behind the cost table (`vq3_A`, `vq3_B`) ran a later build with three performance changes (gather order, a
gather tile cache and fused SpMV row products). Each change was accepted only if the final state after 800 iterations
stayed bitwise equal, so the cost runs inherit this grade; `vq3.status` records the bitwise comparison of both boards'
final states against two earlier builds.

### A known behaviour of the case

On long runs (about 2,200 iterations) HPC_motorbike Small can produce a NaN on the device. The same discretisation in fp64
on the host floors omega in the same cell and fails the same way, so it belongs to the model and mesh, not to fp32 or
the device. tt-fluid stops a run one iteration after any field becomes non-finite.
