# Accuracy and precision

Three questions, three kinds of check:

1. Does the solver discretise like OpenFOAM? The numpy solver is checked against OpenFOAM v2412 field by field.
2. Does the device run the program it was given? The device is compared with the CPU emulator of the same program.
3. Is the fp32 answer good enough? Drag and lift of a motorbike and a car are compared with OpenFOAM in fp64 over windows of iterations.

## 1. The solver against OpenFOAM v2412 (pitzDaily)

The references in `tests/data/pitzDaily_v2412.npz` come from OpenFOAM v2412 runs of the pitzDaily tutorial (k-omega SST, as shipped in `cases/`) with
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
4e-5. SIMPLE carries such differences forward without amplifying them: after 300 iterations, still in the transient,
the device's fields differ from the emulator's by 0.6 % in U, 2.8 % in p and 9 % in omega (max-norm, relative to the
field's largest value), and both converge to the same solution (section 2a). The device itself is deterministic: two runs of the same case
gave bitwise identical fields and histories, and so did the stock wheel and a custom ttnn 0.68.0 build. The emulator is
therefore the reference for what the program computes, and the
answer is graded against OpenFOAM on the device's own output (section 3). The emulator itself is regression-tested in
`tests/test_emulator.py` on one and two chips.

The two partitions give different numbers: after 3 pitzDaily iterations the 1-chip and 2-chip residuals differ by about
1 %. That is fp32 summation order, not an error. Each partition is deterministic, run after run.

## 2a. Converged pitzDaily against OpenFOAM

The comparison that matters is the converged answer. OpenFOAM v2412 (simpleFoam, kOmegaSST, fp64) stops pitzDaily by
residualControl at iteration 381. `ttfluid run cases/pitzDaily -n 1000 --no-check --write DIR` on the same case stops
at iteration 355 on one p300 chip and at 400 on two (AICLK 1350 MHz); the emulator stops at 371. Differences of the
converged cell fields to OpenFOAM, relative to the field's largest value:

| run | U max / rms | p max / rms | k rms | omega rms | nut rms |
|---|---|---|---|---|---|
| device, 1 chip (355 its) | 0.40 % / 0.03 % | 0.76 % / 0.37 % | 0.09 % | 0.11 % | 0.18 % |
| device, 2 chips (400 its) | 0.46 % / 0.05 % | 1.6 % / 0.90 % | 0.16 % | 0.11 % | 0.26 % |
| emulator (371 its) | 0.71 % / 0.09 % | 3.1 % / 1.7 % | 0.24 % | 0.03 % | 0.39 % |

The largest single-cell omega and nut differences (up to 10 % and 9 %) sit in the few near-wall cells where omega is
largest; their rms is the column above. All three runs stop at a residual level of 1e-3, so differences of this size
are what the convergence criterion leaves open, not what the arithmetic adds.

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
transient for every run: tt-fluid is -21.4 counts (2 sigma 11.9), the eight H100 runs -14.7 to +19.3, the B200 +4.2.
On that comparison the Tenstorrent run is the furthest from OpenFOAM. The last line `cdwindow.py` prints applies a
3-count target to it, which this run does not meet.

The eight H100 runs are the same case on identical GPUs, and their window means still spread over 34 counts: that is
how much a window mean moves with nothing but the order of floating-point sums. The Tenstorrent result (-10.3) lies
just outside that spread on the low side and inside OpenFOAM's own +-22.

Reproduce:

```bash
Q=results/tenstorrent/quietbox2-p300x4; O=results/openfoam
python3 bench/nvidia/cdwindow.py --dev $Q/qb2k2_D1A.hist.json:0 --of $O/of1k.dat $O/of4k.dat --from 101 --to 799
python3 bench/nvidia/cdwindow.py --dev $Q/qb2k2_D1A.hist.json:0 $Q/qb2k2_D2A.hist.json:800 --of $O/of1k.dat $O/of4k.dat --from 101 --to 1899
for g in 0 1 2 3 4 5 6 7; do python3 bench/nvidia/grade.py --of $O/of1k.dat $O/of4k.dat --to 800 \
    --run results/nvidia/dgx-h100/A_SPDP_g$g/coefficient.dat; done
```

`cdwindow.py` prints one-sigma errors; `grade.py` prints two-sigma errors. The Tenstorrent history files hold Cd and Cl
of the state each iteration starts from (`Cd_in`), which is OpenFOAM's row for the same iteration number.

The QuietBox 2 runs behind the cost table (`vq3_A`, `vq3_B`) ran a later build with three performance changes (gather order, a
gather tile cache and fused SpMV row products). Each change was accepted only if the final state after 800 iterations
stayed bitwise equal, so the cost runs inherit this grade; `vq3.status` records the bitwise comparison of both boards'
final states against two earlier builds.

## 3a. Grading a new run

`bench/cdgrade.py` turns the comparison above into a pass or fail for any new run of this case, on a device or on a GPU:

```bash
python3 bench/cdgrade.py --run results/tenstorrent/quietbox2-p300x4/qb2k2_D1A.hist.json:0 \
    results/tenstorrent/quietbox2-p300x4/qb2k2_D2A.hist.json:800
python3 bench/cdgrade.py --run results/nvidia/dgx-h100/A_SPDP_g1/coefficient.dat
python3 bench/cdgrade.py --run new_case_1x2_state.npz:0 --base old.hist.json:0     # did a change move the answer?
```

A device run is given as the state file `run_mesh --forces` writes (it holds the history) or as a history exported from
it, followed by the iteration the leg started from.

It compares the mean Cd of a window with OpenFOAM's long-run mean (iterations 501-3372, Cd 0.42226 +- 3.5 counts) and
passes the run if the difference is inside a band. The band is 12.5 counts, or two standard deviations of the
difference where that is wider, and it is known before the run because the noise of a window mean is known:

| measured on | standard deviation of a 100-iteration mean of Cd, counts |
|---|---|
| OpenFOAM v2412 fp64, iterations 201-3372 | 20.7 |
| 18 SPUMA runs, across runs, iterations 301-800 | 16.5 |
| tt-fluid on QuietBox 2, iterations 201-1899 | 15.2 |

The tool uses 20. Blocks of 100 iterations are close to independent, so a window of n blocks has 20 / sqrt(n). The 18
SPUMA runs check this directly: their means over iterations 201-800 have a standard deviation of 8.9 counts, and the
rule gives 8.2.

| window | band, counts | a bias this large fails 9 times in 10 |
|---|---|---|
| 201-800 (one 800-iteration run) | 17.8 | 29 |
| 201-1800 and longer | 12.5 | 20 down to 18 |

Three choices behind it:

- The reference is OpenFOAM's long-run mean, not OpenFOAM over the same iterations. Two runs from the same start are
  unrelated after about 300 iterations, so pairing them only adds the reference's own noise. OpenFOAM's mean over
  101-800 happens to lie 11.8 counts below the average of the 18 SPUMA runs, while that average is 0.7 counts from
  OpenFOAM's long-run mean.
- The window starts at 201. Iterations 101-200 still carry a start-up transient whose size depends on the linear
  solvers (the block mean ranges from -12 to -40 counts across the runs in `results/`).
- An 800-iteration run is a screen for gross errors. It cannot tell a 15-count bias from chance. A change that
  alters the trajectory needs iterations 201-1800 to be held to 12.5 counts.

With `--base` the tool also reports the shift against a baseline run over the same window. Two correct runs that differ
only in the order of their floating-point sums sit 11.5 counts apart (one standard deviation) on 201-800: the 28 pairs
of identical SPUMA SPDP runs measure 16.1, the 28 double-precision pairs 8.8. A shift inside twice that is no change; about 1 pair in 20 of correct runs lies outside.

A change that only reorders the same arithmetic (another partition, another gather or reduction order) has a sharper
check, printed with `--base` when both runs start from the same state. Such runs still track each other early on:
across the 28 pairs of SPUMA SPDP runs Cd differs by at most 0.3 counts through iteration 40 and 5.2 through iteration
100. Double precision against mixed precision is 58 counts apart by then, and a 0.1 % error in the force shows in the
first iterations. A run that tracks its base is a reorder of it; anything else is graded on the window.

What the runs in `results/` grade as:

| run | window | Cd against OpenFOAM, counts | band | grade |
|---|---|---|---|---|
| 16 SPUMA runs on H100, 2 on B200 | 201-800 | -12.6 to +27.0, mean +5.3 | 17.8 | 17 pass, 1 fails (+27.0) |
| tt-fluid, QuietBox 2 board A | 201-800 | -22.8 | 17.8 | fail |
| tt-fluid, QuietBox 2 board A | 201-1899 | -17.0 | 12.5 | fail |

One failure in 18 correct runs is what a two-sigma band gives. The QuietBox 2 run is low in 16 of its 18 blocks, which
is 2.8 standard deviations over 201-1899: either a real low bias of 10 to 17 counts (0.2 to 0.4 % of Cd) or a rare
draw. A 3,000-iteration run will decide; until then this run does not pass.

### A known behaviour of the case

On long runs (about 2,200 iterations) HPC_motorbike Small can produce a NaN on the device. The same discretisation in fp64
on the host floors omega in the same cell and fails the same way, so it belongs to the model and mesh, not to fp32 or
the device. tt-fluid stops a run one iteration after any field becomes non-finite.

## 4. Drag and lift against OpenFOAM (occDrivAer, coarse mesh)

occDrivAer is the DrivAer notchback car from the OpenFOAM HPC Technical Committee with ESI's case setup: 47 wall
patches with wall functions, k-omega SST and a moving-ground slip floor. `cases/occDrivAer/setup.sh` builds the
1,606,238-cell coarse mesh and a potentialFoam start that is bitwise identical to the start of the OpenFOAM reference run.

Like the motorbike, this case does not settle under SIMPLE. OpenFOAM's own Cd has a standard deviation of about 70
counts per iteration over iterations 1001-4000, and its 1000-iteration window means drift from 0.27279 to 0.27172.
The device ran iterations 1-3999 from that start on one Blackhole chip at 1350 MHz (`aiclk.log`).

The first reference was a single serial OpenFOAM v2412 run in fp64. Against it the device reads +15.0 counts over
iterations 1001-3999, a 0.5 % difference, and the gap opens after iteration 2000. To find out whether that is the
device or the flow, OpenFOAM was restarted from its own iteration-1000 fields on 8 MPI ranks, once with the case's own
solver settings (only the domain decomposition changes, so only the order of floating-point sums) and once with linear
solvers like the device's (PCG with a GAMG preconditioner for p, diagonal PBiCGStab for U, k and omega, same
tolerances).

Mean Cd per window in counts against the serial OpenFOAM run over the same iterations (serial means: 0.27279,
0.27232, 0.27174, 0.27228):

| run | 1001-2000 | 2001-3000 | 3001-3999 | 1001-3999 |
|---|---|---|---|---|
| OpenFOAM v2412 fp64, serial | 0 | 0 | 0 | 0 |
| OpenFOAM, 8 ranks, case solvers | +3.6 | +19.5 | +22.8 | +15.1 |
| OpenFOAM, 8 ranks, device-like solvers | +11.2 | +13.1 | +18.2 | +14.0 |
| tt-fluid, one Blackhole chip, fp32 | +3.3 | +20.4 | +21.3 | +15.0 |

One-sigma errors are 4 to 11 counts per 1000-iteration window and about 5 counts over 1001-3999.

OpenFOAM against itself moves as much as the device does, with the same onset after iteration 2000: restarting on 8
ranks is enough to put the flow on a different trajectory. Against the 8-rank OpenFOAM run with the case's solvers the
device is within 1 count in every window (-0.3, +0.9, +1.0; +0.5 over 1001-3999), and within 2.0 counts of the run with
device-like solvers over 1001-3999. So on this case the 3-count target of section 3 is below what OpenFOAM reproduces
between two of its own runs; the device's Cd sits inside that spread. Lift agrees within its noise (+5.5 +- 14.9
counts against serial).

Reproduce:

```bash
T=results/tenstorrent/occdrivaer-coarse; O=results/openfoam/occdrivaer-coarse
for ref in serial p8 p8dev; do
  python3 bench/nvidia/cdwindow.py --dev $T/dev.hist.json:0 --of $O/$ref.dat --from 1001 --to 3999
done
```

The device run was split into checkpointed segments on two chips; a second chip run with different segment lengths
is bitwise identical to it through iteration 3199, so restarts are exact.
