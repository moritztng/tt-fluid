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

HPC_motorbike Small (8.66 M cells, k-omega SST) does not settle to a steady state under SIMPLE: OpenFOAM's own drag
coefficient has a standard deviation of 42 drag counts from iteration to iteration (1 count = 0.0001 in Cd), and two
runs from the same start are unrelated after about 300 iterations. Single iterations cannot be compared. A run is
graded on its mean Cd over a window of iterations against OpenFOAM v2412 in fp64 on a CPU, from the same potentialFoam
start (`results/openfoam/of1k.dat`, `of4k.dat`).

`bench/cdgrade.py` does this for any run of the case, on a device or on a GPU, and exits 0 on a pass:

```bash
python3 bench/cdgrade.py --run results/tenstorrent/quietbox2-p300x4/qb2k2_D1A.hist.json:0 \
    results/tenstorrent/quietbox2-p300x4/qb2k2_D2A.hist.json:800
python3 bench/cdgrade.py --run results/nvidia/dgx-h100/A_SPDP_g1/coefficient.dat
python3 bench/cdgrade.py --run new_case_1x2_state.npz:0 --base old.hist.json:0     # did a change move the answer?
```

A device run is given as the state file `run_mesh --forces` writes (it holds the history) or as a history exported from
it, followed by the iteration the leg started from. The history holds Cd and Cl of the state each iteration starts
from (`Cd_in`), which is OpenFOAM's row for the same iteration number.

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

The rule stops holding on longer windows of one run. The long tt-fluid histories on record (a 5,000-iteration run on a
Wormhole Galaxy, 3,000 on four p150a cards, 1,900 on QuietBox 2) give seven windows of 1,100 to 1,900 iterations, and
their means spread 10.5 counts where independent blocks would give 5. The Wormhole Galaxy run reads +6.5 counts on
201-1800 and -16.0 on 1801-3400. So the tool counts one run as 600 iterations at most, however long its window, and the 12.5 bar is
reached by pooling runs (below), not by running one of them longer.

| graded | band, counts | a bias this large fails 9 times in 10 |
|---|---|---|
| one run, window 201-800 or longer | 17.8 | 29 |
| four or more runs pooled | 12.5 | 19 |

Three choices behind it:

- The reference is OpenFOAM's long-run mean, not OpenFOAM over the same iterations. Two runs from the same start are
  unrelated after about 300 iterations, so pairing them only adds the reference's own noise. OpenFOAM's mean over
  101-800 happens to lie 11.8 counts below the average of the 18 SPUMA runs, while that average is 0.7 counts from
  OpenFOAM's long-run mean.
- The window starts at 201. Iterations 101-200 still carry a start-up transient whose size depends on the linear
  solvers (the block mean ranges from -12 to -40 counts across the runs in `results/`).
- One run is a screen for gross errors. It cannot tell a 15-count bias from chance. A change that alters the
  trajectory is held to 12.5 counts on four or more runs pooled.

With `--base` the tool also reports the shift against a baseline run over the same window. Two correct runs that differ
only in the order of their floating-point sums sit 11.5 counts apart (one standard deviation) on 201-800: the 28 pairs
of identical SPUMA SPDP runs measure 16.1, the 28 double-precision pairs 8.8. A shift inside twice that is no change;
about 1 pair in 20 of correct runs lies outside.

A change that only reorders the same arithmetic (another partition, another gather or reduction order) has a sharper
check, printed with `--base` when both runs start from the same state. Such runs still track each other through
iteration 10: two partitions of the same device program differ by at most 0.5 counts of Cd there, the 28 pairs of
SPUMA SPDP runs by 0.01. The limit is 1 count. Double precision against mixed precision is 58 counts apart by then, and
a 0.1 % error in the force is 17 counts at iteration 1. The check stops at iteration 10 because the device pair
separates soon after: at iteration 12 one velocity solve takes 14 iterations on one partition and 13 on the other, and
three iterations later the two drag values are 44 counts apart. A run that tracks its base is a reorder of it; anything
else is graded on the window. Eleven iterations are enough for this check: given a run that short, the tool prints
the tracking line alone.

Runs that track each other are draws of one arithmetic, so their windows can be pooled into one grade, which is how
a set of 800-iteration runs reaches the 12.5 bar. Each `--pool` adds a run; the tool refuses one that does not track
the first, and it leaves out the first block of every added run because that block still follows the first run. The
pooled error is the larger of the rule's and the one the run means themselves show, so runs that disagree widen their
own band. Four runs or more make a grade of record. Such runs cost nothing extra where several cases run at once:
start each from a state one rounding step away from the others. The eight SPUMA SPDP runs on H100 and the one on B200
pool to +8.4 +- 5.0 counts over 5,400 iterations: a pass.

```bash
H=results/nvidia/dgx-h100
python3 bench/cdgrade.py --run $H/A_SPDP_g0/coefficient.dat --pool results/nvidia/b200/A_SPDP_g0/coefficient.dat \
    $(for g in 1 2 3 4 5 6 7; do echo --pool $H/A_SPDP_g$g/coefficient.dat; done)
```

What the runs in `results/` grade as:

| runs | window | Cd against OpenFOAM, counts | band | grade |
|---|---|---|---|---|
| SPUMA SPDP, 8 runs on H100 and 1 on B200, pooled | 201-800 | +8.4 +- 5.0 | 12.5 | pass |
| 16 SPUMA runs on H100 and 2 on B200 (SPDP and fp64), one by one | 201-800 | -12.6 to +27.0, mean +5.3 | 17.8 | 17 pass, 1 fails (+27.0) |
| tt-fluid, QuietBox 2 board A, 2 chips | 201-1899 | -17.0 | 17.8 | pass, by 0.8 |
| tt-fluid, Blackhole Galaxy, 2 chips | 201-1999 | +3.8 | 17.8 | pass |
| tt-fluid, Blackhole Galaxy, 4 chips | 201-799 | -4.3 | 17.8 | pass |
| the three tt-fluid runs pooled | as above | -6.0 +- 8.4 | 16.9 | pass (three runs: a screen) |

One failure in 18 correct runs is what a two-sigma band gives. The three tt-fluid runs are one fp32 program on three
partitions of the mesh (`results/tenstorrent/galaxy-blackhole/runs.status` has chips, commit, clock and final state of
the Galaxy legs). They track each other to 0.5 counts through iteration 10 and then separate like any two runs. Taken
alone, iterations 201-800 of the QuietBox 2 run read -22.8, outside the band; the pool reads -6.0 +- 8.4. The run
means spread 7.7 counts where the rule gives 5.0, so the pool carries its own wider band, and with three runs it is a
screen. A grade of record at 12.5 needs a fourth run.

```bash
G=results/tenstorrent/galaxy-blackhole; Q=results/tenstorrent/quietbox2-p300x4
python3 bench/cdgrade.py --run $G/hC_1.hist.json:0 $G/hC_2.hist.json:500 $G/hC_3.hist.json:1000 $G/hC_4.hist.json:1500 \
    --pool $Q/qb2k2_D1A.hist.json:0 $Q/qb2k2_D2A.hist.json:800 --pool $G/d_l0_b0.hist.json:0
```

The QuietBox 2 runs behind the cost table (`vq3_A`, `vq3_B`) ran a later build with three performance changes (gather
order, a gather tile cache and fused SpMV row products). Each change was accepted only if the final state after 800
iterations stayed bitwise equal, so the cost runs follow the board A history above; `vq3.status` records the bitwise
comparison of both boards' final states against two earlier builds.

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
device-like solvers over 1001-3999. Two OpenFOAM runs of this case differ by 15 counts, so nothing below that can be
asked of the device; its Cd sits inside that spread. Lift agrees within its noise (+5.5 +- 15.8 counts against serial).

Reproduce:

```bash
T=results/tenstorrent/occdrivaer-coarse; O=results/openfoam/occdrivaer-coarse
for ref in serial p8 p8dev; do
  python3 bench/cdgrade.py --means --run $T/dev.hist.json:0 --ref $O/$ref.dat --from 1001 --to 3999
done
```

`--means` prints the window means of both runs with their batch errors and no grade: the band of section 3 is measured
on the motorbike and does not carry over to another case.

The device run was split into checkpointed segments on two chips; a second chip run with different segment lengths
is bitwise identical to it through iteration 3199, so restarts are exact.
