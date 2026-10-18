# Performance methodology

The question this page answers is narrow: **what does one HPC_motorbike Small case cost on each system**, when
each system runs the same physics from the same start and is graded against the same OpenFOAM reference? Everything
below is computed by `bench/per_dollar.py` from the files in `results/`.

## The case

HPC_motorbike Small from the OpenFOAM HPC Technical Committee benchmark set
(`incompressible/simpleFoam/HPC_motorbike/Small`: OpenFOAM's motorBike geometry meshed with snappyHexMesh to
8,662,309 cells), steady incompressible
simpleFoam with k-omega SST, 800 SIMPLE iterations from a potentialFoam initialisation, the solver settings the case
ships with (p: tolerance 1e-7, relTol 0.01; U, k, omega: tolerance 1e-8, relTol 0.1). Both sides run all 800
iterations; neither stops early. Accuracy is graded on the drag and lift history ([accuracy.md](accuracy.md)).

## Cost per case

One formula for every system, the usual three-year total cost of ownership:

```
$/h    = capex / (3 x 8760 h)  +  wall kW x PUE 1.3 x $0.15 per kWh
$/case = batch wall time x $/h / 3600 / cases in the batch
```

A batch fills the box with independent cases: one case per GPU on an 8-GPU server, one case per p300c board (2 chips)
on a QuietBox 2. The batch wall time is that of the slowest member, so the box is charged for every second any part of
it is busy.

| | QuietBox 2 | 8x H100 SXM | 8x B200 |
|---|---|---|---|
| capex | $9,999 | $249,200 | $384,228 |
| price source | Tenstorrent store, host included, 2026-10-16 | ASUS ESC N8-E11 (HGX H100) at SHI, 2026-10-15 (MSRP $257,651) | Supermicro SYS-A22GA-NBRT-G1 (HGX B200) at Newegg, last listed price, 2026-09-21 |
| wall kW | 1.30 (system rating, not measured) | 3.13 | 3.68 |
| how kW is derived | full rating charged | measured busy board power 226.8 W per GPU / 0.94 PSU + 0.15 kW host per GPU | measured busy board power 291.9 W / 0.94 + 0.15 kW host |
| $/h | 0.634 | 10.093 | 15.339 |
| cases in the batch | 2 | 8 | 8 |
| batch wall | 1310.3 s (slower board, process wall) | 771.5 s (makespan of 8 concurrent runs) | 690.5 s (one GPU; 8x assumed) |
| $/case | **0.1154** | 0.2704 | 0.3678 |
| QuietBox 2 advantage per $ | | **2.34x** | **3.19x** |

The B200 row is derived. One case ran on one B200; the 8-GPU batch is assumed to take the same time, which flatters
the B200 (the measured H100 batch of 8 took 4 % longer than its fastest member).

## A fair baseline

The GPU side is [SPUMA](https://arxiv.org/abs/2512.22215), the CUDA port of OpenFOAM v2412, the strongest public GPU
OpenFOAM we know of. The rules we held ourselves to:

- **Fastest measured precision mode.** SPUMA ran both in fp64 (DP) and in OpenFOAM's mixed mode (SPDP: fp32 fields
  and matrices, fp64 sums in the linear solvers). SPDP was faster on both GPUs (H100 batch 771.5 s vs 831.6 s, B200
  690.5 s vs 740.6 s) and is the one in the table.
- **The GPU-tuned solver settings.** The pressure solver uses CINECA's GPU-tuned GAMG settings
  (`bench/nvidia/fvSolution.spuma.mbs`) at the case's tolerances.
- **Less work for the GPU, not more.** Only the force-coefficient function object runs on the GPU side; the case's
  probes and other function objects are switched off there.
- **A correctness fix in the GPU's favour.** SPUMA's device min/max reduction misses negative cells, so OpenFOAM's
  `bound()` stops firing and the SST model turns a negative omega into NaN. We fixed it
  (`bench/nvidia/boundfix/boundFix.C`, loaded with `LD_PRELOAD`) rather than report a failed GPU run.
- **Cheapest documented price.** NVIDIA gets the lowest server listing we could document. Tenstorrent gets its own
  store price, which a buyer would negotiate down, and its full power rating.
- **One case per GPU.** The case fits easily in one GPU's memory, and splitting one case over several GPUs costs
  efficiency. Packing two cases on one GPU could raise GPU throughput; we did not measure it, and a bound derived from
  GPU utilisation would leave the QuietBox 2 ahead at about 1.35x.

Build and run scripts: `bench/nvidia/spuma_build.sh`, `spdp_build.sh`, `spuma_pack.sh`, `spuma_min.sh`,
`spuma_run.sh`, `boundfix.sh`.

## Hardware and software

| | Tenstorrent | NVIDIA |
|---|---|---|
| system | QuietBox 2: 2x p300c (4 Blackhole chips), AMD Ryzen 7 9700X host, 256 GB | 8x H100 80GB HBM3 SXM (700 W limit), 2x Xeon Platinum 8480+; 1x B200 |
| clocks | AICLK 1350 MHz on all four chips for the whole run (`vq3_aiclk.log`, every 10 s) | GPU clocks as reported by nvidia-smi (`gpu.csv`, every second) |
| software | ttnn 0.68.0, tt-fluid at this release's device program | SPUMA (OpenFOAM v2412 GPU port), NVHPC, driver 615.71.09 |
| precision | fp32 throughout | SPDP mixed |

## What the number is not

- It is not a speed record. One H100 finishes a case in about 750 s; one p300c board takes about 1300 s. The advantage
  comes from the price of the box.
- It is one case. HPC_motorbike Small is a realistic snappyHexMesh external-aerodynamics mesh, but other meshes and models will land
  elsewhere.
- Larger Tenstorrent systems (the 32-chip Galaxy) and the 65 M-cell occDrivAer car case are on the roadmap. We do not
  quote numbers for them here because their raw data is not in this release.
