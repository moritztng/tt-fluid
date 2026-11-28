# HPC_motorbike Small on two chips of a Blackhole Galaxy: the tutorial's run line

One case on two neighbouring chips (`--mesh 1,2`), stock ttnn 0.68.0 wheel, 2026-11-28. These legs back the device
times in the [tutorial](../../../docs/tutorial.md) and the changelog. They are two chips of a Galaxy, not a
QuietBox 2: the cost table in the README does not use them.

| leg | run line | iterations | device time | run wall | process wall | host peak |
|---|---|---|---|---|---|---|
| `C.100` | the line before (`--l1 1 --gcache 14`) | 100 | 161.15 s | 164.39 s | 234.6 s | 31.1 GB |
| `S.d100` | `--l1 2 --l1-bytes 1461760 --lsplit 1 --gcache 32 --bpatch 4` | 100 | 136.01 s | 138.43 s | 227.2 s | 28.6 GB |
| `S2.d100` | the same line, images read by stored bytes only | 100 | 136.01 s | 138.81 s | 159.3 s | 11.3 GB |
| `S.800` | as `S.d100`, forces on | 800 | 1,031.5 s | 1,031.9 s | 1,141.1 s | 31.0 GB |
| `XC.1` | as `S2.d100`, the day's control for the two legs below | 100 | 136.01 s | 136.11 s | 156.4 s | 11.9 GB |
| `X63.1` | as `XC.1` with `--bgts 6 --tgts 3` | 100 | 133.52 s | 137.67 s | 170.9 s | 11.9 GB |
| `B3.1` | the tutorial's line, with the shares given by hand: `--l1 1 --l1-bytes 1461760 --lsplit 2 --cse 2 --tabdense 530 --span 2 --bgts 6 --tgts 3` | 100 | 108.18 s | 112.20 s | 140.7 s | 14.4 GB |
| `M3.1` | the same line on the build with the fp32 omega floor (2ab0309) | 100 | 107.99 s | 111.97 s | 139.2 s | 11.9 GB |
| `M3d.1` | the tutorial's line as it is printed, no share option given, on the same build | 100 | 107.96 s | 108.03 s | 129.2 s | 11.9 GB |
| `B352.1` | `B3.1`'s line with `--bgts 5 --tgts 2` | 100 | 112.41 s | 116.25 s | 142.0 s | 14.4 GB |

- Device time is core 0's cycle count over the iterations at 1350 MHz (`run 0 wall ... ticks [...]` in each log). The
  process wall includes the image upload, the state download and the save.
- 161.15 s / 136.01 s = 1.18, 161.15 s / 108.18 s = 1.49 and 136.01 s / 108.18 s = 1.26. The saved states of
  `C.100`, `S.d100`, `S2.d100`, `XC.1`, `X63.1` and `B3.1` are bit for bit the same file contents (compared outside
  this repository; the states are 1 GB each and not stored here). Their residual and force histories are here and
  identical: `*.hist.json`.
- `M3.1` runs the build that floors omega at 1e-6 in fp32 (changelog, Fixed). That moves the rounding of every fp32
  run from its first iterations, so its history differs from the others in the last digits. The options of the line
  are the same as in `B3.1`, and the device time is within 0.2 %.
- `M3d.1` gives neither `--bgts` nor `--tgts`. Its log has the line `--bgts / --tgts auto: 6 / 3`, its state and
  history are `M3.1`'s bit for bit, and its device time is 0.02 % from `M3.1`'s.
- `B352.1` is `B3.1` with the shares the tutorial gave before. 112.41 s / 108.18 s = 1.04, and the state is the
  same.
- `*.clk`: one line per second with the time, `in` or `out` of the iterations, the host load and the AICLK of both
  chips in MHz, read from sysfs while the leg ran. In the iterations every sample is 1343 to 1350 MHz.
- `S.800.cd` is the output of `python3 bench/cdgrade.py --run S.800.hist.json:0 --from 201`: Cd is 9.6 counts above
  OpenFOAM fp64 on iterations 201-799, inside the one-run band of 17.8. The grade of record is the pool in
  [docs/accuracy.md](../../../docs/accuracy.md), not this one run.
- Home paths in the logs are shortened to `~`. Nothing else is edited.
