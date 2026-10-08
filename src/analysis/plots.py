"""Scatter of external AD-vs-CN AUC against cohort-classification AUC (figure of the first submission, now AUC):
marker = FM, colour = source (adaptation) cohort, shade = target cohort, dashed line = OLS over all points."""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

import common as C

MARKERS = {"brainiac": "o", "sammed": "*", "3dino": "D", "medicalsam2": "P", "curia2_cls": "^", "nnfoundation_vit": "v"}
BASE_COLORS = {"AIBL": "#1f77b4", "BrainLAT": "#ff7f0e", "OASIS3": "#2ca02c", "OASIS4": "#f2e76e",
               "NigerianBrain": "#d62728", "ADNI": "#9467bd"}
LIGHTEN = [0.0, 0.1, 0.3, 0.5, 0.7]  # one shade per target cohort (5 targets per source)


def _shade(color, amount):
    c = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(c + (1 - c) * amount)



# ---------------------------------------------------------------------------------------------------------------------
# Paper figures are drawn at their printed size (Elsevier cas-dc: text width 494.5 pt = 6.84 in, column width
# 238.3 pt = 3.30 in) so that the font sizes below are the sizes in the PDF. Include them in LaTeX with
# width=\textwidth (full-width figures) or width=\linewidth (single-column figures).
# ---------------------------------------------------------------------------------------------------------------------
FULL_W, COL_W = 6.84, 3.30
PAPER_RC = {
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.title_fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
}
COHORT_LABELS = {"ADNI": "ADNI", "AIBL": "AIBL", "BrainLAT": "BrainLAT", "NigerianBrain": "Nigerian Brain",
                 "OASIS3": "OASIS-3", "OASIS4": "OASIS-4"}
MINUS = "\N{MINUS SIGN}"


def _num(x, fmt=".2f"):
    """Number with a typographic minus sign."""
    return format(x, fmt).replace("-", MINUS)


def _save(fig, path):
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{path}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def pair_scatter(d, path, line=None, title=None):
    """External AD-vs-CN AUC against cohort-classification AUC for every (FM, adaptation, test) cell.
    Marker = FM, colour = adaptation cohort, shade = test cohort. ``line`` = (intercept, slope) of the LMM fixed
    effect, drawn as a dashed line; if None, an OLS line over all cells is drawn instead."""
    shade = {}
    for src in C.COHORTS:
        targets = sorted(t for t in C.COHORTS if t != src)
        for t, a in zip(targets, LIGHTEN):
            shade[(src, t)] = _shade(BASE_COLORS[src], a)
    models = [m for m in C.FMS if m in set(d["model"])]
    sources = [c for c in C.COHORTS if c in set(d["train"])]
    msize = {"sammed": 5.5}

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(FULL_W, 4.8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.55], hspace=0.2, left=0.075, right=0.995, top=0.99,
                              bottom=0.01)
        ax, lax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        for r in d.sort_values(["train", "test"]).itertuples():
            ax.plot(r.auc_cohort_mean, r.auc_disease_mean, marker=MARKERS[r.model], color=shade[(r.train, r.test)],
                    markersize=msize.get(r.model, 4.2), markeredgecolor="#4d4d4d", markeredgewidth=0.25,
                    alpha=0.9, linestyle="none")
        x, y = d["auc_cohort_mean"].to_numpy(), d["auc_disease_mean"].to_numpy()
        a, b = line if line is not None else np.polyfit(x, y, 1)[::-1]
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, a + b * xs, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Cohort separability (cohort classification AUC, adaptation vs. test cohort)")
        ax.set_ylabel("External AD classification AUC")
        if title:
            ax.set_title(title)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(True, color="#e6e6e6", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(x.min() - 0.03, x.max() + 0.03)
        ax.set_ylim(y.min() - 0.03, y.max() + 0.03)

        lax.axis("off")
        leg1 = lax.legend(handles=[Line2D([0], [0], marker=MARKERS[m], color="#333333", linestyle="none",
                                    markersize=msize.get(m, 4.2) + 0.5, label=C.FM_LABELS[m]) for m in models],
                          title="Foundation model", loc="upper center", bbox_to_anchor=(0.5, 1.04),
                          ncol=len(models), frameon=False, columnspacing=1.6, handletextpad=0.3)
        lax.add_artist(leg1)
        # one legend column per adaptation cohort: bold header, then its five test cohorts in their shades
        handles, bold = [], []
        for src in sources:
            bold.append(len(handles))
            handles.append(Line2D([0], [0], linestyle="none", label=COHORT_LABELS[src]))
            for t in sorted(t for t in C.COHORTS if t != src and (src, t) in shade):
                handles.append(Line2D([0], [0], marker="o", color=shade[(src, t)], markeredgecolor="#4d4d4d",
                                      markeredgewidth=0.25, linestyle="none", markersize=4.2,
                                      label=f"\N{RIGHTWARDS ARROW} {COHORT_LABELS[t]}"))
        leg = lax.legend(handles=handles, title="Adaptation cohort \N{RIGHTWARDS ARROW} test cohort",
                         loc="lower center", bbox_to_anchor=(0.5, -0.04),
                          ncol=len(sources), frameon=False, columnspacing=1.4, handletextpad=0.2, handlelength=1.0,
                          labelspacing=0.25)
        for i in bold:
            leg.get_texts()[i].set_fontweight("bold")
        _save(fig, path)


def pair_scatter_abstract(d, path, annotation=None):
    """Compact version of `pair_scatter` for the graphical abstract: small canvas with large fonts, colour = adaptation
    cohort only (no per-target shade), and two short legends side by side below the axes instead of the 30-entry pair legend."""
    cohorts = [c for c in C.COHORTS if c in set(d["train"])]
    models = [m for m in C.FMS if m in set(d["model"])]

    fig, ax = plt.subplots(figsize=(10, 7))
    for r in d.itertuples():
        ax.plot(r.auc_cohort_mean, r.auc_disease_mean, marker=MARKERS[r.model], color=BASE_COLORS[r.train],
                markersize=11, alpha=0.75, markeredgecolor="white", markeredgewidth=0.6, linestyle="none")
    x, y = d["auc_cohort_mean"].to_numpy(), d["auc_disease_mean"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, a + b * xs, color="black", linestyle="--", linewidth=3.5)
    if annotation:
        ax.text(0.03, 0.04, annotation, transform=ax.transAxes, ha="left", va="bottom", fontsize=18, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", linewidth=1.5))

    ax.set_xlabel("Cohort separability in features\n(cohort classification AUC)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Generalization\n(AD vs. CN AUC on test cohort)", fontsize=20, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_linewidth(2.2)
    ax.tick_params(width=2.2, length=6, labelsize=17)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.2)
    ax.set_xlim(x.min() - 0.03, x.max() + 0.03)
    ax.set_ylim(y.min() - 0.03, y.max() + 0.03)

    leg1 = ax.legend(handles=[Line2D([0], [0], marker="o", color=BASE_COLORS[c], linestyle="none", markersize=14,
                                     label=c.replace("NigerianBrain", "Nigerian Brain")) for c in cohorts],
                     title="Adaptation cohort", loc="upper center", bbox_to_anchor=(0.27, -0.21),
                     fontsize=16, title_fontsize=17, frameon=False, handletextpad=0.2, alignment="left")
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=[Line2D([0], [0], marker=MARKERS[m], color="black", linestyle="none", markersize=14,
                                     label=C.FM_LABELS[m]) for m in models],
                     title="Foundation model", loc="upper center", bbox_to_anchor=(0.73, -0.21),
                     fontsize=16, title_fontsize=17, frameon=False, handletextpad=0.2, alignment="left")
    leg2.get_title().set_fontweight("bold")
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    fig.savefig(f"{path}.svg", bbox_inches="tight")
    plt.close(fig)


INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, MEAN = "#e1e0d9", "#c3c2b7", "#d9d7cf"


def internal_external(d, path, title=None):
    """Per FM: internal (held-out split of the adaptation cohort) vs external (unseen test cohorts) AD-vs-CN AUC.
    Lines pair each adaptation cohort's internal AUC with its mean external AUC (colour = adaptation cohort), small dots
    = single test cohorts, the wide light band = mean over adaptation cohorts; the mean drop is printed above each FM."""
    models = [m for m in C.FMS if m in set(d["model"])]
    cohorts = [c for c in C.COHORTS if c in set(d["train"])]
    rng = np.random.default_rng(0)
    xi, xe = -0.2, 0.2

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(FULL_W, 2.9))
        gs = fig.add_gridspec(1, 2, width_ratios=[5.55, 1.0], wspace=0.02, left=0.065, right=0.995, top=0.99,
                              bottom=0.13)
        ax, lax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        for i, m in enumerate(models):
            dm = d[d["model"] == m]
            per = dm.groupby("train").agg(internal=("auc_disease_internal_mean", "first"),
                                          external=("auc_disease_mean", "mean"))
            mi, me = per["internal"].mean(), per["external"].mean()
            ax.plot([i + xi, i + xe], [mi, me], color=MEAN, lw=7, solid_capstyle="round", zorder=1)
            ax.scatter(i + xe + 0.09 + rng.uniform(-0.03, 0.03, len(dm)), dm["auc_disease_mean"], s=4, alpha=0.5,
                       color=[BASE_COLORS[t] for t in dm["train"]], edgecolor="white", lw=0.2, zorder=2)
            for c, r in per.iterrows():
                ax.plot([i + xi, i + xe], [r.internal, r.external], color=BASE_COLORS[c], lw=1.0, marker="o",
                        markersize=3.2, markeredgecolor="white", markeredgewidth=0.6, solid_capstyle="round",
                        zorder=3)
            ax.text(i, 0.995, f"\N{GREEK CAPITAL LETTER DELTA} {_num(me - mi, '+.3f')}", ha="center", va="top",
                    fontsize=7, color=INK_2)

        ax.set_xticks([i + x for i in range(len(models)) for x in (xi, xe)])
        ax.set_xticklabels(["Internal", "External"] * len(models), fontsize=6.5, color=INK_MUTED)
        for i, m in enumerate(models):
            ax.text(i, -0.1, C.FM_LABELS[m], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7,
                    fontweight="semibold", color=INK)
        for i in range(1, len(models)):
            ax.axvline(i - 0.5, color=GRID, lw=0.6, zorder=0)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_ylim(min(0.45, d["auc_disease_mean"].min() - 0.02), 1.0)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.set_ylabel("AD classification AUC", color=INK_2)
        if title:
            ax.set_title(title, fontweight="semibold", color=INK)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(axis="y", labelcolor=INK_MUTED, length=0)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)

        lax.axis("off")
        handles = [Line2D([0], [0], color=BASE_COLORS[c], marker="o", lw=1.0, markersize=3.2, markeredgecolor="white",
                          markeredgewidth=0.6, label=COHORT_LABELS[c]) for c in cohorts]
        handles.append(Line2D([0], [0], color=MEAN, lw=6, solid_capstyle="round", label="Mean over cohorts"))
        leg = lax.legend(handles=handles, title="Adaptation cohort", loc="center left", frameon=False,
                         alignment="left", labelcolor=INK_2, handlelength=1.6)
        leg.get_title().set_color(INK)
        _save(fig, path)


CONDITION_COLORS = ["#0072B2", "#D55E00", "#009E73"]


def condition_figure(conds: dict, path, title=None):
    """Compare feature conditions (ComBat variants or pooling methods) in three panels.

    conds: {label: (cells DataFrame, LMM slope of cohort AUC)} - first entry is the reference (mean pooling, original).
    (a) cohort-classification AUC per FM, (b) external AD-vs-CN AUC per FM: small dots = cells (source->target pairs),
    large diamonds = mean over cells. (c) external vs cohort AUC for every cell with an OLS line per condition; the
    legend of (c) gives the LMM slope of each condition.
    """
    labels = list(conds)
    models = [m for m in C.FMS if m in set(next(iter(conds.values()))[0]["model"])]
    k = len(labels)
    off = (np.arange(k) - (k - 1) / 2) * 0.26
    rng = np.random.default_rng(0)

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(FULL_W, 2.75))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.95], height_ratios=[1, 0.08], wspace=0.36, hspace=0.75,
                              left=0.065, right=0.995, top=0.93, bottom=0.01)
        axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
        lax = fig.add_subplot(gs[1, :])
        for ax, col, ylab in zip(axes[:2], ["auc_cohort_mean", "auc_disease_mean"],
                                 ["Cohort classification AUC", "External AD classification AUC"]):
            for j, lab in enumerate(labels):
                d = conds[lab][0]
                for i, m in enumerate(models):
                    v = d.loc[d["model"] == m, col].to_numpy()
                    ax.scatter(i + off[j] + rng.uniform(-0.06, 0.06, len(v)), v, s=2.5, alpha=0.35,
                               color=CONDITION_COLORS[j], lw=0)
                    ax.scatter(i + off[j], v.mean(), s=16, marker="D", color=CONDITION_COLORS[j], edgecolor="black",
                               lw=0.4, zorder=3, label=lab if i == 0 else None)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([C.FM_LABELS[m] for m in models], rotation=35, ha="right", rotation_mode="anchor",
                               fontsize=6.5)
            ax.set_xlim(-0.6, len(models) - 0.4)
            ax.set_ylabel(ylab)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.grid(axis="y", color="#e6e6e6", lw=0.5)
            ax.set_axisbelow(True)
        ax = axes[2]
        for j, lab in enumerate(labels):
            d, beta = conds[lab]
            x, y = d["auc_cohort_mean"], d["auc_disease_mean"]
            ax.scatter(x, y, s=2.5, alpha=0.4, color=CONDITION_COLORS[j], lw=0)
            b, a = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, a + b * xs, color=CONDITION_COLORS[j], lw=1.4, label=fr"$\beta_1$ = {_num(beta)}")
        ax.set_xlabel("Cohort classification AUC")
        ax.set_ylabel("External AD classification AUC")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
        leg = ax.legend(title="LMM slope", loc="lower left", frameon=True, framealpha=0.85, edgecolor="none",
                        handlelength=1.4, borderaxespad=0.2, borderpad=0.3, fontsize=6.5, title_fontsize=6.5)
        leg.get_frame().set_linewidth(0)
        for ax, t in zip(axes, ["(a) Cohort separability", "(b) External AD classification", "(c) All pairs"]):
            ax.set_title(t, loc="left", fontweight="bold", pad=4)
        lax.axis("off")
        h, l = axes[0].get_legend_handles_labels()
        lax.legend(h, l, loc="center", ncol=k, frameon=False, columnspacing=2.0, handletextpad=0.3)
        if title:
            fig.suptitle(title)
        _save(fig, path)


def mantel_null(null, rho_bar, path):
    """Permutation distribution of the FM-averaged within-source Spearman correlation (03_mantel_test): all
    relabellings, those with a statistic <= the observed one (one-sided, H1: negative association) in red."""
    eps = 1e-12
    n_lower = int((null <= rho_bar + eps).sum())
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(COL_W, 1.85))
        bins = np.arange(-0.6, 0.6 + 1e-9, 0.025)
        ax.hist(null, bins=bins, color="#9cc0ee", edgecolor="white", linewidth=0.3, label="All relabellings")
        ax.hist(null[null <= rho_bar + eps], bins=bins, color="#d9534f", edgecolor="white", linewidth=0.3,
                label=fr"$\bar\rho \leq$ observed ({n_lower}/{len(null)})")
        ax.axvline(rho_bar, color="black", lw=1.0, label=fr"Observed $\bar\rho$ = {_num(rho_bar)}")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, 1.65 * np.histogram(null, bins=bins)[0].max())  # headroom for the legend
        ax.set_xlabel(r"$\bar\rho$ under cohort relabelling")
        ax.set_ylabel("Number of relabellings")
        ax.legend(frameon=False, loc="upper right", fontsize=6.5, handlelength=1.2, borderaxespad=0.1)
        fig.tight_layout(pad=0.2)
        _save(fig, path)
