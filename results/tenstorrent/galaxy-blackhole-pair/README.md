# HPC_motorbike Small on two chips of a Blackhole Galaxy: the tutorial's run line

One case on two neighbouring chips (`--mesh 1,2`), stock ttnn 0.68.0 wheel, 2026-11-28. These legs back the
"1.18 times faster" and "about 1030 s" statements in the [tutorial](../../../docs/tutorial.md) and the changelog.
They are two chips of a Galaxy, not a QuietBox 2: the cost table in the README does not use them.

| leg | run line | iterations | device time | run wall | process wall | host peak |
|---|---|---|---|---|---|---|
| `C.100` | the line before (`--l1 1 --gcache 14`) | 100 | 161.15 s | 164.39 s | 234.6 s | 31.1 GB |
| `S.d100` | `--l1 2 --l1-bytes 1461760 --lsplit 1 --gcache 32 --bpatch 4` | 100 | 136.01 s | 138.43 s | 227.2 s | 28.6 GB |
| `S2.d100` | the same line, images read by stored bytes only | 100 | 136.01 s | 138.81 s | 159.3 s | 11.3 GB |
| `S.800` | as `S.d100`, forces on | 800 | 1,031.5 s | 1,031.9 s | 1,141.1 s | 31.0 GB |

- Device time is core 0's cycle count over the iterations at 1350 MHz (`run 0 wall ... ticks [...]` in each log). The
  process wall includes the image upload, the state download and the save.
- 161.15 s / 136.01 s = 1.18. The saved states of `C.100`, `S.d100` and `S2.d100` are bit for bit the same file
  contents (compared outside this repository; the states are 1 GB each and not stored here). Their residual and force
  histories are here and identical: `*.hist.json`.
- `*.clk`: one line per second with the time, `in` or `out` of the iterations, the host load and the AICLK of both
  chips in MHz, read from sysfs while the leg ran. In the iterations every sample is 1343 to 1350 MHz.
- `S.800.cd` is the output of `python3 bench/cdgrade.py --run S.800.hist.json:0 --from 201`: Cd is 9.6 counts above
  OpenFOAM fp64 on iterations 201-799, inside the one-run band of 17.8. The grade of record is the pool in
  [docs/accuracy.md](../../../docs/accuracy.md), not this one run.
- Home paths in the logs are shortened to `~`. Nothing else is edited.
