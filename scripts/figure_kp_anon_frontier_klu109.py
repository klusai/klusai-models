#!/usr/bin/env python3
"""KLU-109 — render the committed privacy-utility (Pareto) frontier figure from the scorecard.

Reads ``docs/klu-109-scorecard.json`` and emits a dependency-free **SVG** (no matplotlib —
the program avoids heavy plotting deps) plotting, per frontier language, the redaction baseline vs
kp-anon on the privacy-utility plane:

  * x-axis = **utility** = ``1 − structural_disruption.mask_token_ratio`` (↑ better, right = more
    usable document),
  * y-axis = **re-identification leak** = ``redaction_leakage.leak_rate`` (↓ better, lower = safer).

The desirable region is the lower-right (high utility, low leak). The figure shows kp-anon sitting
far to the RIGHT of the redaction baseline (much higher utility — no mask tokens) but slightly
HIGHER on the leak axis (~2-3% vs the blanket-mask baseline's 0%, since substitution can only
replace what the shared detector finds): it trades a small, bounded privacy cost for a large utility
gain and does NOT strictly dominate (headline: 0/2 langs). Per-seed kp-anon points show the >=2-seed spread.

The figure is labelled config_status=dev with NO SOTA/validated claim (KLU-109 / KLU-27).

    python scripts/figure_kp_anon_frontier_klu109.py \
        --scorecard docs/klu-109-scorecard.json \
        --out docs/klu-109-kp-anon-frontier.svg
"""

from __future__ import annotations

import json
from pathlib import Path

import click

W, H = 720, 460
M_L, M_R, M_T, M_B = 70, 200, 56, 64       # margins (room on the right for a legend)
PLOT_W = W - M_L - M_R
PLOT_H = H - M_T - M_B
LANG_COLORS = {"ro": "#1f77b4", "pl": "#d62728", "en": "#2ca02c", "it": "#9467bd"}


def _x(util: float) -> float:
    return M_L + util * PLOT_W           # utility 0..1 → left..right


def _y(leak: float) -> float:
    return M_T + (1.0 - leak) * PLOT_H   # leak 0 at bottom (safe), 1 at top (leaky)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg(cells: list[dict], headline: dict) -> str:
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">')
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    # Title.
    parts.append(f'<text x="{W/2}" y="24" text-anchor="middle" font-size="16" font-weight="bold">'
                 'kp-anon vs redaction baseline — privacy-utility frontier (Track C)</text>')
    parts.append(f'<text x="{W/2}" y="42" text-anchor="middle" font-size="11" fill="#555">'
                 'config_status=dev · held-out real-skeleton · NO SOTA/validated claim (gated KLU-27)'
                 '</text>')
    # Axes.
    parts.append(f'<line x1="{M_L}" y1="{M_T}" x2="{M_L}" y2="{M_T+PLOT_H}" stroke="#333"/>')
    parts.append(f'<line x1="{M_L}" y1="{M_T+PLOT_H}" x2="{M_L+PLOT_W}" y2="{M_T+PLOT_H}" stroke="#333"/>')
    # Gridlines + ticks.
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = _x(f)
        parts.append(f'<line x1="{gx}" y1="{M_T}" x2="{gx}" y2="{M_T+PLOT_H}" stroke="#eee"/>')
        parts.append(f'<text x="{gx}" y="{M_T+PLOT_H+16}" text-anchor="middle" font-size="10" '
                     f'fill="#555">{f:.2f}</text>')
        gy = _y(f)
        parts.append(f'<line x1="{M_L}" y1="{gy}" x2="{M_L+PLOT_W}" y2="{gy}" stroke="#eee"/>')
        parts.append(f'<text x="{M_L-8}" y="{gy+3}" text-anchor="end" font-size="10" '
                     f'fill="#555">{f:.2f}</text>')
    # Axis labels.
    parts.append(f'<text x="{M_L+PLOT_W/2}" y="{H-22}" text-anchor="middle" font-size="12">'
                 'utility = 1 − mask-token ratio  (→ more usable)</text>')
    parts.append(f'<text x="20" y="{M_T+PLOT_H/2}" text-anchor="middle" font-size="12" '
                 f'transform="rotate(-90 20 {M_T+PLOT_H/2})">re-id leak rate  (↓ safer)</text>')
    # "better" arrow toward lower-right.
    parts.append(f'<text x="{M_L+PLOT_W-4}" y="{M_T+PLOT_H-6}" text-anchor="end" font-size="10" '
                 f'fill="#2ca02c">↘ better (high utility, low leak)</text>')

    ly = M_T + 6
    for cell in cells:
        if cell.get("status") and cell.get("status") != "scored":
            continue
        langs = cell.get("languages") or ["?"]
        lang = langs[0]
        color = LANG_COLORS.get(lang, "#444")
        seeds = cell.get("seeds", [])
        if not seeds:
            continue
        # Baseline point (same across seeds in leak; mask-ratio identical too) — use seed 0.
        b = seeds[0]
        bx, by = _x(1.0 - b["mask_token_ratio_baseline"]), _y(b["leak_rate_baseline"])
        # kp-anon point(s) — one per seed (show spread).
        anon_pts = [(_x(1.0 - s["mask_token_ratio_kp_anon"]), _y(s["leak_rate_kp_anon"])) for s in seeds]
        ax = sum(p[0] for p in anon_pts) / len(anon_pts)
        ay = sum(p[1] for p in anon_pts) / len(anon_pts)
        # Dominance arrow baseline → kp-anon.
        parts.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" '
                     f'stroke="{color}" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.7"/>')
        # Baseline marker = hollow square.
        parts.append(f'<rect x="{bx-5:.1f}" y="{by-5:.1f}" width="10" height="10" fill="white" '
                     f'stroke="{color}" stroke-width="2"/>')
        # kp-anon markers = filled circles (per seed) + a larger mean circle.
        for px, py in anon_pts:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" opacity="0.55"/>')
        parts.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{ax+9:.1f}" y="{ay+3:.1f}" font-size="11" fill="{color}" '
                     f'font-weight="bold">{_esc(lang.upper())}</text>')

    # Legend box.
    lx = M_L + PLOT_W + 18
    parts.append(f'<text x="{lx}" y="{ly}" font-size="11" font-weight="bold">legend</text>')
    ly += 18
    parts.append(f'<rect x="{lx}" y="{ly-9}" width="10" height="10" fill="white" stroke="#444" '
                 f'stroke-width="2"/>')
    parts.append(f'<text x="{lx+16}" y="{ly}" font-size="10">redaction baseline</text>')
    ly += 18
    parts.append(f'<circle cx="{lx+5}" cy="{ly-3}" r="5" fill="#444"/>')
    parts.append(f'<text x="{lx+16}" y="{ly}" font-size="10">kp-anon</text>')
    ly += 22
    for lang in sorted({(c.get("languages") or ["?"])[0] for c in cells if c.get("seeds")}):
        parts.append(f'<circle cx="{lx+5}" cy="{ly-3}" r="5" fill="{LANG_COLORS.get(lang,"#444")}"/>')
        parts.append(f'<text x="{lx+16}" y="{ly}" font-size="10">{_esc(lang.upper())}</text>')
        ly += 16
    ly += 8
    dom = headline.get("n_languages_kp_anon_dominates_redaction_baseline_all_seeds", 0)
    nlang = len(headline.get("frontier_languages_scored", []))
    parts.append(f'<text x="{lx}" y="{ly}" font-size="10" fill="#333">dominates baseline:</text>')
    ly += 14
    parts.append(f'<text x="{lx}" y="{ly}" font-size="10" fill="#333">{dom}/{nlang} langs (all seeds)</text>')

    parts.append('</svg>')
    return "\n".join(parts)


@click.command()
@click.option("--scorecard", default="docs/klu-109-scorecard.json")
@click.option("--out", default="docs/klu-109-kp-anon-frontier.svg")
def main(scorecard, out):
    sc = json.loads(Path(scorecard).read_text())
    svg = _svg(sc["frontier_cells"], sc.get("headline", {}))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(svg, encoding="utf-8")
    click.echo(f"wrote frontier figure -> {out}")


if __name__ == "__main__":
    main()
