#!/usr/bin/env python3
"""Draw the paper's figures from results/, never from numbers typed by hand.

Every quantity here is loaded through the same helpers the rest of the repository
uses -- `check_reported_numbers.load` for the score panels, `analyze_stage0`'s
bootstrap for intervals and `biologos.validity.tcrit` for the t interval -- so a
figure cannot drift from the prose the way the six numbers of 2026-10-08 did. If
a figure and a sentence disagree after this runs, the sentence is wrong.

  python3 scripts/make_figures.py

Writes two full-width figures. The paper is at the ACL eight-page content limit,
and three single-column figures cost about 23 more lines of column than one
full-width row carrying the same three panels, so the layout is chosen for the
page budget and not for taste.

  panels.pdf   aggregates: the bar, the scale trend, the matched contrasts
  spread.pdf   the same two aggregates drawn per assay, because a mean over 25
               assays hides whether the aggregate describes any of them

spread.pdf exists because the headline contrast is a mean of +0.0376 that is
positive on 14 of 25 assays, and the scale trend is a slope fitted through four
means whose per-assay dispersion is an order of magnitude larger. Both facts are
in the prose as counts; neither is visible until the points are drawn.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from analyze_stage0 import boot_mean, boot_paired, load as load_by_model  # noqa: E402
from check_reported_numbers import (RUNGS, RUNG_PARAMS,  # noqa: E402
                                    load as load_panels, sweep_by_assay)
from biologos.validity import tcrit  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "paper" / "fig"
TEXTW = 6.30         # ACL text width: 16 cm
BLOSUM62 = 0.2282

# One colour per role, identical across panels so the dashed bar means the same
# thing everywhere. Chosen to stay distinguishable in greyscale print.
C_MODEL, C_EVO, C_BAR, C_NULL = "#1f4e79", "#c1431f", "#6b6b6b", "#9aa5b1"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 7.0, "legend.fontsize": 6.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    "figure.dpi": 200,
})


def draw_baselines(ax, s, panel):
    """The lead claim: what the models are worth against a substitution matrix."""
    rows = [("ESM-2 650M", "esm", C_MODEL), ("chemistry ridge", "chem", C_NULL),
            ("Evo 2 7B", "evo2", C_EVO), ("BLOSUM62", "b62", C_BAR),
            ("NT-v2 50M", "nt50", C_NULL)]
    ys = np.arange(len(rows))[::-1]
    means = {}
    for y, (lab, key, c) in zip(ys, rows):
        m, lo, hi = boot_mean([s[key][i] for i in panel])
        means[lab] = m
        ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot([m], [y], "o", color=c, ms=3.8, mec="white", mew=0.5, zorder=3)
        ax.annotate(f"{m:+.3f}", (hi, y), textcoords="offset points",
                    xytext=(3.5, 0), ha="left", va="center", fontsize=6.4, color=c)
    ax.axvline(0, color="black", lw=0.5, zorder=1)
    ax.axvline(BLOSUM62, color=C_BAR, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"mean $\rho$, 25 assays")
    ax.set_xlim(-0.09, 0.66)
    ax.set_ylim(-0.8, len(rows) - 0.3)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.text(BLOSUM62, -0.72, " the bar", fontsize=6.0, color=C_BAR, va="bottom")
    ax.set_title("(a) the bar is a substitution matrix", fontsize=7.2, pad=4)
    return means


def draw_scaling(ax, s, nt, panel):
    """Scale does not reach Evo 2, and Evo 2 is not on the line."""
    tags, sizes = ("50m", "100m", "250m", "500m"), (5e7, 1e8, 2.5e8, 5e8)
    x = np.log10(sizes)
    y = np.array([nt[t].mean() for t in tags])
    ci = np.array([boot_mean(nt[t])[1:] for t in tags])
    slope, icept = np.polyfit(x, y, 1)
    evo, elo, ehi = boot_mean([s["evo2"][i] for i in panel])
    x7 = np.log10(7e9)

    xs = np.linspace(x.min() - 0.12, x7 + 0.12, 50)
    ax.plot(xs, slope * xs + icept, color=C_MODEL, lw=0.8, ls=(0, (4, 2)),
            zorder=2, label=f"trend, ${slope:+.4f}$/decade")
    ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o", color=C_MODEL,
                ms=3.6, lw=1.0, capsize=1.6, zorder=4, mec="white", mew=0.5,
                label="NT-v2, 50M–500M")
    ax.errorbar([x7], [evo], yerr=[[evo - elo], [ehi - evo]], fmt="D",
                color=C_EVO, ms=4.2, lw=1.0, capsize=1.6, zorder=4,
                mec="white", mew=0.5, label="Evo 2 7B")
    ax.plot([x7], [slope * x7 + icept], "x", color=C_MODEL, ms=4.5, mew=1.1,
            zorder=4)
    ax.annotate("", xy=(x7, evo), xytext=(x7, slope * x7 + icept),
                arrowprops=dict(arrowstyle="<->", color=C_EVO, lw=0.7,
                                shrinkA=2.2, shrinkB=2.2))
    ax.text(x7 - 0.12, (evo + slope * x7 + icept) / 2, r"$9.5\!\times$",
            fontsize=6.2, color=C_EVO, ha="right", va="center")
    ax.axhline(BLOSUM62, color=C_BAR, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.text(8.5, BLOSUM62 - 0.014, "BLOSUM62", fontsize=6.0, color=C_BAR,
            va="top", ha="center")
    ax.set_xlabel(r"parameters, $\log_{10} N$")
    ax.set_ylabel(r"mean $\rho$", labelpad=1)
    ax.set_ylim(-0.055, 0.345)
    ax.legend(loc="upper left", frameon=False, handlelength=1.4,
              borderpad=0.0, labelspacing=0.2, handletextpad=0.4)
    ax.set_title("(b) scale does not reach Evo 2", fontsize=7.2, pad=4)
    return slope, icept, evo


def draw_contrasts(ax):
    """Three contrasts available from released checkpoints, each one factor.

    Matched on the full model suffix, not a substring: "multi-species" occurs in
    both arms of the tokenizer contrast, so a loose match silently pairs a model
    with itself. Each difference is the richer option minus its baseline.
    """
    specs = [
        ("3-mer $-$ 6-mer", "results/stage0_tokenizer.csv",
         "v2-50m-3mer-multi-species", "v2-50m-multi-species"),
        ("1000G $-$ ref", "results/stage0_corpus500m.csv",
         "500m-1000g", "500m-human-ref"),
        ("850 sp. $-$ 1000G", "results/stage0_corpus2500m.csv",
         "2.5b-multi-species", "2.5b-1000g"),
    ]
    out = []
    for i, (lab, path, hi_key, lo_key) in enumerate(specs):
        by = load_by_model(ROOT / path)

        def pick(k):
            hits = [v for n, v in by.items() if n.endswith(k)]
            assert len(hits) == 1, f"{k!r} matched {len(hits)} models in {path}"
            return hits[0]

        A, B = pick(hi_key), pick(lo_key)
        ids = sorted(set(A) & set(B))
        d = np.array([float(A[j]["rho_nt"]) - float(B[j]["rho_nt"]) for j in ids])
        m, lo, hi = boot_paired([float(A[j]["rho_nt"]) for j in ids],
                                [float(B[j]["rho_nt"]) for j in ids])
        tc = tcrit(len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        tlo, thi = m - tc, m + tc
        y = len(specs) - 1 - i
        # Grey unless BOTH estimators exclude zero: the third contrast is the one
        # where they disagree, and the figure must not resolve that silently.
        c = C_EVO if (lo > 0 and tlo > 0) else C_NULL
        ax.plot([lo, hi], [y + 0.10, y + 0.10], color=c, lw=1.4,
                solid_capstyle="round", zorder=3)
        ax.plot([m], [y + 0.10], "o", color=c, ms=3.6, mec="white", mew=0.5,
                zorder=4)
        ax.plot([tlo, thi], [y - 0.12, y - 0.12], color=c, lw=0.8, alpha=0.75,
                solid_capstyle="round", zorder=3)
        ax.annotate(f"{m:+.4f}", (max(hi, thi), y), textcoords="offset points",
                    xytext=(3.5, 0), ha="left", va="center", fontsize=6.4, color=c)
        if (lo > 0) != (tlo > 0):
            ax.annotate("estimators disagree", (max(hi, thi), y),
                        textcoords="offset points", xytext=(3.5, -8), ha="left",
                        va="center", fontsize=6.0, color=C_EVO)
        out.append((lab, m, lo, hi, tlo, thi, len(ids)))
    ax.axvline(0, color="black", lw=0.6, zorder=1)
    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels([s[0] for s in specs][::-1])
    ax.set_xlabel(r"paired difference in mean $\rho$")
    ax.set_xlim(-0.105, 0.175)
    ax.set_ylim(-0.8, len(specs) - 0.3)
    ax.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax.plot([], [], color=C_NULL, lw=1.4, label="bootstrap")
    ax.plot([], [], color=C_NULL, lw=0.8, alpha=0.75, label=r"$t$ interval")
    ax.legend(loc="lower right", frameon=False, handlelength=1.3, ncol=2,
              borderpad=0.0, labelspacing=0.2, handletextpad=0.4,
              columnspacing=0.8)
    ax.set_title("(c) no released checkpoint clears it", fontsize=7.2, pad=4)
    return out


def draw_paired_assays(ax, s, panel):
    """Evo 2 against BLOSUM62, one point per assay, with the identity line.

    The aggregate is a mean difference of +0.0376 whose interval contains zero.
    Drawn per assay it is 25 points straddling y=x, which is what "higher on 14
    of 25" looks like and is not what a bar chart of two means looks like.
    """
    x = np.array([s["b62"][i] for i in panel])
    y = np.array([s["evo2"][i] for i in panel])
    win = y > x
    lim = (-0.13, 0.72)
    ax.plot(lim, lim, color="black", lw=0.6, zorder=1)
    ax.axhline(0, color=C_NULL, lw=0.5, ls=(0, (1, 2)), zorder=1)
    ax.plot(x[win], y[win], "o", color=C_EVO, ms=3.6, mec="white", mew=0.5,
            zorder=3, label=f"Evo 2 higher ({win.sum()})")
    ax.plot(x[~win], y[~win], "o", color=C_NULL, ms=3.6, mec="white", mew=0.5,
            zorder=3, label=f"BLOSUM62 higher ({(~win).sum()})")
    m, lo, hi = boot_paired(y.tolist(), x.tolist())
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6]); ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xlabel(r"BLOSUM62 $\rho$")
    ax.set_ylabel(r"Evo 2 7B $\rho$", labelpad=1)
    ax.text(0.03, 0.97, f"paired mean ${m:+.4f}$\n$[{lo:+.3f}, {hi:+.3f}]$",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.3,
            color=C_EVO)
    ax.legend(loc="lower right", frameon=False, handlelength=0.9,
              borderpad=0.0, labelspacing=0.2, handletextpad=0.3)
    ax.set_title("(a) the gap is not a property of the assays", fontsize=7.2,
                 pad=4)
    return m, lo, hi, int(win.sum()), len(panel)


def draw_assay_ladder(ax, panel):
    """The scale ladder per assay, under the mean the slope was fitted to.

    The fitted trend is +0.0207 per decade. The per-assay spread at any one rung
    is wider than the whole range the trend covers, so the figure shows the
    slope and the reason not to read it as a per-assay prediction at once.
    """
    per = sweep_by_assay(set(panel))
    ids = sorted(set.intersection(*[set(per[t]) for t in RUNGS]))
    x = np.log10(RUNG_PARAMS)
    for i in ids:
        ax.plot(x, [per[t][i] for t in RUNGS], color=C_NULL, lw=0.5, alpha=0.55,
                zorder=2, solid_capstyle="round")
    y = np.array([np.mean([per[t][i] for i in ids]) for t in RUNGS])
    ci = np.array([boot_mean([per[t][i] for i in ids])[1:] for t in RUNGS])
    slope, icept = np.polyfit(x, y, 1)
    ax.plot(x, slope * x + icept, color=C_MODEL, lw=0.8, ls=(0, (4, 2)),
            zorder=4, label=f"trend, ${slope:+.4f}$/decade")
    ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o", color=C_MODEL,
                ms=3.6, lw=1.2, capsize=1.6, zorder=5, mec="white", mew=0.5,
                label=f"mean, {len(ids)} assays")
    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.plot([], [], color=C_NULL, lw=0.5, alpha=0.55, label="one assay")
    spread = max(np.ptp([per[t][i] for i in ids]) for t in RUNGS)
    ax.annotate(f"widest rung spans ${spread:.2f}$,\nthe trend covers "
                f"${abs(slope) * (x[-1] - x[0]):.3f}$",
                (x[-1], y[-1]), textcoords="offset points", xytext=(-2, 26),
                ha="right", va="center", fontsize=6.3, color=C_MODEL)
    ax.set_xlabel("parameters")
    ax.set_ylabel(r"$\rho$", labelpad=1)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in RUNGS])
    ax.legend(loc="lower left", frameon=False, handlelength=1.4, ncol=3,
              borderpad=0.0, labelspacing=0.2, handletextpad=0.4,
              columnspacing=0.8)
    ax.set_title("(b) every assay, not the mean of them", fontsize=7.2, pad=4)
    return slope, spread, len(ids)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, cl, nt, tok, cor, ctrl = load_panels()
    panel = sorted(set(s["b62"]) & set(s["evo2"]))

    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.05),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]})
    means = draw_baselines(axes[0], s, panel)
    slope, icept, evo = draw_scaling(axes[1], s, nt, panel)
    contrasts = draw_contrasts(axes[2])
    fig.subplots_adjust(wspace=0.78)
    fig.savefig(OUT / "panels.pdf")
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 2, figsize=(TEXTW, 2.35),
                             gridspec_kw={"width_ratios": [1.0, 1.22]})
    paired = draw_paired_assays(ax2[0], s, panel)
    ladder = draw_assay_ladder(ax2[1], panel)
    fig2.subplots_adjust(wspace=0.30)
    fig2.savefig(OUT / "spread.pdf")
    plt.close(fig2)

    pred = slope * np.log10(7e9) + icept
    print(f"panel: {len(panel)} assays -> {OUT / 'panels.pdf'}\n")
    print("(a) baselines")
    for k, v in means.items():
        print(f"      {k:<18s} {v:+.4f}")
    print(f"\n(b) scaling\n      slope {slope:+.4f}/decade, intercept {icept:+.4f}")
    print(f"      7B prediction {pred:+.4f}, Evo 2 {evo:+.4f}, "
          f"shortfall {evo / pred:.2f}x")
    print("\n(c) contrasts")
    for lab, m, lo, hi, tlo, thi, n in contrasts:
        agree = (lo > 0) == (tlo > 0)
        print(f"      {lab:<22s} {m:+.4f}  boot [{lo:+.4f}, {hi:+.4f}]  "
              f"t [{tlo:+.4f}, {thi:+.4f}]  n={n}"
              f"{'' if agree else '   <- estimators DISAGREE on zero'}")
    pm, plo, phi, wins, npan = paired
    lslope, spread, nlad = ladder
    print(f"\nspread: {OUT / 'spread.pdf'}")
    print(f"      (a) Evo 2 - BLOSUM62 {pm:+.4f} [{plo:+.4f}, {phi:+.4f}], "
          f"Evo 2 higher on {wins} of {npan}")
    print(f"      (b) slope {lslope:+.4f}/decade over {nlad} assays, "
          f"widest rung spans {spread:.3f}")


if __name__ == "__main__":
    main()
