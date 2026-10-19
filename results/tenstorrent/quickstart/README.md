# pitzDaily on one Blackhole p300 (1 and 2 chips)

Result rows of `ttfluid run` for this release, stock ttnn 0.68.0 wheel, AICLK 1350 MHz (recorded in each row).

| file | command | stopped at |
|---|---|---|
| `f1.jsonl` | `ttfluid run cases/pitzDaily -n 1000 --chips 1 --no-check --write DIR` | 355 (residualControl) |
| `f2.jsonl` | `ttfluid run cases/pitzDaily -n 1000 --chips 2 --no-check` | 400 (residualControl) |
| `q1.jsonl` | `ttfluid run cases/pitzDaily -n 300 --chips 1` (with the emulator check, `vs_emu`) | 300 |
| `q2.jsonl` | `ttfluid run cases/pitzDaily -n 300 --chips 2` (with the emulator check, `vs_emu`) | 300 |

The converged fields' comparison with OpenFOAM v2412 is in [docs/accuracy.md](../../../docs/accuracy.md).
