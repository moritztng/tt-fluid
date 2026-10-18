"""Draw docs/figures/headline.svg from the raw data: cost per case (bench/per_dollar.py) and the drag coefficient of
every run against OpenFOAM fp64 over iterations 101-800 (the window of docs/accuracy.md).

    python3 docs/figures/make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "bench" / "nvidia")]
import per_dollar  # noqa: E402
from cdwindow import batch, of_series  # noqa: E402

R = ROOT / "results"
LO, HI = 101, 800


def cd_mean(series):
    return float(np.mean([series[i][0] for i in range(LO, HI + 1)]))


def data():
    rows = per_dollar.systems(R)
    for s in rows:
        s["usd_h"] = per_dollar.tco_h(s["capex"], s["kw"])
        s["usd_case"] = s["s_batch"] / 3600 * s["usd_h"] / s["cases"]
    of = of_series([R / "openfoam/of1k.dat", R / "openfoam/of4k.dat"])
    of_cd = cd_mean(of)
    _, of_se, _ = batch([of[i][0] for i in range(LO, HI + 1)], 100)
    hist = json.load(open(R / "tenstorrent/quietbox2-p300x4/qb2k2_D1A.hist.json"))
    tt = {1 + k: (r["Cd_in"], r["Cl_in"]) for k, r in enumerate(hist)}
    h100 = [cd_mean(of_series([R / f"nvidia/dgx-h100/A_SPDP_g{g}/coefficient.dat"])) for g in range(8)]
    b200 = cd_mean(of_series([R / "nvidia/b200/A_SPDP_g0/coefficient.dat"]))
    cnt = lambda x: (x - of_cd) * 1e4
    return rows, dict(band=2e4 * of_se, tt=cnt(cd_mean(tt)), h100=[cnt(x) for x in h100], b200=cnt(b200))


STYLE = """<style>
  .s { --bg:#fcfcfb; --t1:#0b0b0b; --t2:#52514e; --t3:#8a8984; --grid:#e4e3df; --band:#ecebe7; --c1:#2a78d6; --c2:#eb6834; }
  @media (prefers-color-scheme: dark) {
    .s { --bg:#1a1a19; --t1:#f0efec; --t2:#c3c2b7; --t3:#8f8e88; --grid:#33332f; --band:#2a2a27; --c1:#3987e5; --c2:#d95926; }
  }
  .s text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: var(--t1); }
  .h { font-size: 15px; font-weight: 600; } .sub { font-size: 12px; fill: var(--t2) !important; }
  .l { font-size: 12px; } .v { font-size: 12px; font-weight: 600; } .ax { font-size: 11px; fill: var(--t3) !important; }
</style>"""


def bar_path(x0, y, w, h, r=4):
    """Horizontal bar from the baseline x0, square at the baseline, 4px rounded at the data end."""
    r = min(r, w, h / 2)
    return (f"M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} Q{x0 + w:.1f},{y:.1f} {x0 + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x0 + w:.1f},{y + h:.1f} {x0 + w - r:.1f},{y + h:.1f} H{x0:.1f} Z")


def svg(rows, cd):
    W, H = 1040, 330
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         'aria-label="Cost per case and drag coefficient versus OpenFOAM">', STYLE, '<g class="s">',
         f'<rect width="{W}" height="{H}" rx="8" fill="var(--bg)"/>']
    # left: $/case
    names = ["QuietBox 2 (4 Blackhole chips)", "8x H100 SXM", "8x B200 (derived)"]
    x0, x1, top = 210, 450, 92
    vmax = 0.40
    o.append('<text class="h" x="24" y="34">Cost per case, HPC_motorbike Small (8.66 M cells)</text>')
    o.append('<text class="sub" x="24" y="54">US$ per 800-iteration case, 3-year TCO; lower is better</text>')
    for t in (0, 0.1, 0.2, 0.3, 0.4):
        x = x0 + (x1 - x0) * t / vmax
        o.append(f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{top + 3 * 56 - 26}" stroke="var(--grid)"/>')
        o.append(f'<text class="ax" x="{x:.1f}" y="{top + 3 * 56 - 10}" text-anchor="middle">{t:.1f}</text>')
    for i, (s, n) in enumerate(zip(rows, names)):
        y = top + i * 56
        w = (x1 - x0) * s["usd_case"] / vmax
        o.append(f'<text class="l" x="{x0 - 10}" y="{y + 15}" text-anchor="end">{n}</text>')
        o.append(f'<path d="{bar_path(x0, y, w, 22)}" fill="var(--{"c1" if i == 0 else "c2"})"/>')
        lab = f'${s["usd_case"]:.3f}' + ("" if i == 0 else f'  ({s["usd_case"] / rows[0]["usd_case"]:.2f}x the cost)')
        o.append(f'<text class="v" x="{x0 + w + 8:.1f}" y="{y + 15}">{lab}</text>')
    o.append(f'<text class="ax" x="24" y="{H - 40}">bench/per_dollar.py on results/: batch wall x $/h / cases per batch.</text>')
    o.append(f'<text class="ax" x="24" y="{H - 24}">Blue: Tenstorrent, fp32. Orange: NVIDIA, SPUMA mixed precision.</text>')
    # right: Cd vs OpenFOAM
    X0, X1, lo, hi = 700, 900, -40, 40
    sx = lambda v: X0 + (X1 - X0) * (v - lo) / (hi - lo)
    o.append(f'<text class="h" x="{X0 - 160}" y="34">Drag coefficient vs OpenFOAM fp64</text>')
    o.append(f'<text class="sub" x="{X0 - 160}" y="54">mean over iterations {LO}-{HI}, drag counts (0.0001)</text>')
    yb, ye = 80, 240
    o.append(f'<rect x="{sx(-cd["band"]):.1f}" y="{yb}" width="{sx(cd["band"]) - sx(-cd["band"]):.1f}" '
             f'height="{ye - yb}" fill="var(--band)"/>')
    for t in range(lo, hi + 1, 20):
        o.append(f'<line x1="{sx(t):.1f}" y1="{yb}" x2="{sx(t):.1f}" y2="{ye}" stroke="var(--grid)"/>')
        o.append(f'<text class="ax" x="{sx(t):.1f}" y="{ye + 16}" text-anchor="middle">{t:+d}</text>'.replace("+0", "0"))
    o.append(f'<line x1="{sx(0):.1f}" y1="{yb}" x2="{sx(0):.1f}" y2="{ye}" stroke="var(--t3)" stroke-width="1.5"/>')
    o.append(f'<text class="ax" x="{sx(0) + 4:.1f}" y="{yb + 12}">OpenFOAM</text>')
    for j, (lab, vals, c) in enumerate((("tt-fluid, QuietBox 2", [cd["tt"]], "c1"),
                                        ("8x H100, 8 runs", cd["h100"], "c2"), ("B200", [cd["b200"]], "c2"))):
        y = yb + 34 + j * 46
        o.append(f'<text class="l" x="{X0 - 10}" y="{y + 4}" text-anchor="end">{lab}</text>')
        for v in vals:
            o.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="5" fill="var(--{c})" stroke="var(--bg)" stroke-width="2"/>')
        txt = f"{vals[0]:+.1f}" if len(vals) == 1 else f"{min(vals):+.1f} to {max(vals):+.1f}"
        o.append(f'<text class="v" x="{X1 + 12}" y="{y + 4}">{txt}</text>')
    o.append(f'<text class="ax" x="{X0 - 160}" y="{H - 40}">bench/nvidia/cdwindow.py and grade.py on results/; same '
             'potentialFoam start.</text>')
    o.append(f'<text class="ax" x="{X0 - 160}" y="{H - 24}">Shaded: +-{cd["band"]:.0f} counts, the 2-sigma error of '
             "OpenFOAM's own window mean.</text>")
    o.append("</g></svg>")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    rows, cd = data()
    out = ROOT / "docs/figures/headline.svg"
    out.write_text(svg(rows, cd))
    print(out, {k: (round(v, 1) if isinstance(v, float) else [round(x, 1) for x in v]) for k, v in cd.items()})
