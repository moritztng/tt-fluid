# Block A/B (flx-kernel): generated SFPU bodies (`--blocks 1`) against stock op words (`--blocks 0`)

Run on the box from `~/flx/kernel` with the tree in `~/flx/kernel/tree` and these files in `~/flx/kernel/run`.

* `pzprep.sh`, `hprep.sh`: lower once (mcache), then write the images of arm A, arm B and the broken arm X
  (`--blocks 2`: the first add of every body subtracts) for pitzDaily and for the hood on one chip. CPU only.
* `chainB.sh`, `chainP.sh`: the device legs on one chip, A0 B0 (discarded) A1 B1 A2 B2 X, each under the chip's
  flock, with the HANDBACK, window, holder and closed-node checks of `ablib.sh` and a sysfs AICLK sampler.
* `absum.py STATUS PREFIX`: the leg lines with the in-run clock, the means, and the digest comparison.
* `blkreal.py MCACHE`: the instruction count of the real generated programs per tile-it, priced with the unit costs.
* `rejects.py MCACHE`: the bodies the generator turns down, and why.
