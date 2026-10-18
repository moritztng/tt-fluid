"""Cost per case of HPC_motorbike Small (8.66 M cells, k-omega SST, 800 SIMPLE iterations) on each system, from the raw
run data in results/ and the list prices below. Prints a markdown table; `--json` writes the same numbers.

    python3 bench/per_dollar.py [--results results] [--json out.json]

Method (the same formula for every system):

    $/h    = capex / (3 years x 8760 h) + wall kW x PUE 1.3 x $0.15/kWh
    $/case = s/case x $/h / 3600 / cases the box runs at once

s/case is the wall time of a batch that fills the box (8 cases, one per GPU, on 8 GPUs; one case per p300c board on a
QuietBox 2), taken from the slowest member. GPU wall power is the measured mean board power while the GPUs were busy,
divided by a 0.94 power supply efficiency, plus 0.15 kW of host per GPU. The QuietBox 2 is charged its full 1.3 kW
rating, not a measurement. NVIDIA gets the cheapest documented price for each system; see docs/performance.md.
"""
import argparse
import glob
import json
import re
from pathlib import Path

PUE, USD_KWH, HOURS = 1.3, 0.15, 3 * 8760
PSU, GPU_HOST_KW = 0.94, 0.15


def tco_h(capex, kw):
    return capex / HOURS + kw * PUE * USD_KWH


def makespan(plan, tag):
    m = re.findall(rf"\] {tag} makespan ([\d.]+) s", Path(plan).read_text())
    return float(m[-1])


def busy_board_kw(pattern):
    """Mean nvidia-smi board power over the samples with nonzero utilisation, all files matched."""
    w = []
    for f in glob.glob(pattern):
        for line in open(f):
            p = [x.strip() for x in line.split(",")]
            if int(p[5].rstrip(" %")) > 0:
                w.append(float(p[2].rstrip(" W")))
    return sum(w) / len(w) / 1e3


def process_wall(log):
    return float(re.search(r"process wall ([\d.]+) s", Path(log).read_text()).group(1))


def systems(r):
    r = Path(r)
    h100_kw = busy_board_kw(str(r / "nvidia/dgx-h100/A_SPDP_g*/gpu.csv"))
    b200_kw = busy_board_kw(str(r / "nvidia/b200/A_SPDP_g0/gpu.csv"))
    qb = r / "tenstorrent/quietbox2-p300x4"
    return [
        dict(name="Tenstorrent QuietBox 2 (2x p300c, 4 Blackhole chips)", capex=9_999, kw=1.30, cases=2,
             s_batch=max(process_wall(qb / "vq3_A.log"), process_wall(qb / "vq3_B.log")),
             how="measured: one case per board, both boards at once, slower board's process wall",
             price="Tenstorrent store, host included"),
        dict(name="8x H100 SXM (HGX)", capex=249_200, kw=8 * (h100_kw / PSU + GPU_HOST_KW), cases=8,
             s_batch=makespan(r / "nvidia/dgx-h100/plan.out", "A SPDP"), board_kw=h100_kw,
             how="measured: 8 cases, one per GPU, SPUMA SPDP, makespan", price="cheapest documented HGX H100 listing"),
        dict(name="8x B200 (HGX)", capex=384_228, kw=8 * (b200_kw / PSU + GPU_HOST_KW), cases=8,
             s_batch=makespan(r / "nvidia/b200/plan.out", "A SPDP"), board_kw=b200_kw,
             how="derived: one case on one B200 measured, 8 GPUs assumed to run 8 such cases in the same time",
             price="cheapest documented HGX B200 listing"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = systems(a.results)
    for s in rows:
        s["usd_h"] = tco_h(s["capex"], s["kw"])
        s["s_case"] = s["s_batch"] / s["cases"]
        s["usd_case"] = s["s_batch"] / 3600 * s["usd_h"] / s["cases"]
    tt = rows[0]["usd_case"]
    print("| system | capex $ | wall kW | $/h | cases at once | batch wall s | s/case | $/case | Tenstorrent per $ advantage | how |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        s["tt_advantage"] = s["usd_case"] / tt
        print(f"| {s['name']} | {s['capex']:,} | {s['kw']:.2f} | {s['usd_h']:.3f} | {s['cases']} | {s['s_batch']:.1f} | "
              f"{s['s_case']:.1f} | {s['usd_case']:.4f} | {s['tt_advantage']:.2f}x | {s['how']} |")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
