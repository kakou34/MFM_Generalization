"""06 - Feature-space analysis: t-SNE (qualitative) and 2-Wasserstein distances (quantitative).

Mean-pooled features of the six FMs (3DINO original: ``feats.csv``), for the original features (main text) and
the ComBat-harmonised features (sex; sex + age; supplement, files get the suffix ``_combat_sex`` / ``_combat_sex_age``).

t-SNE        : one embedding per FM on all scans. Rows: diagnosis | cohort | scanner manufacturer |
               field strength (colour = category in every row). Main figure = the --n-main FMs with the best mean external AUC,
               the other FMs go to tsne_supplement_*.
2-Wasserstein: distance between every (cohort, diagnosis) group, ``wasserstein_rel`` as defined in
               ``adaptation/baseline_preds.py``:  (W2^2(A,B) - E_perm[W2^2]) / (trS_A + trS_B),
               i.e. noise-floor corrected and scale free, so FMs share one colour scale. Exact OT (POT),
               each group capped at N_PER_GROUP scans drawn with a fixed seed.
Quantitative summary: ratio of the between-cohort distance (same diagnosis) to the CN-vs-AD distance within cohort,
               per FM, with and without Nigerian Brain (>1 = cohort differences exceed disease differences).

Usage:  python 06_feature_space_figures.py [--fms 3dino brainiac ...] [--skip-tsne] [--skip-w2] [--skip-combat]
Needs:  pot (pip install pot), scikit-learn, seaborn
"""
from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Polygon

import common as C

N_PER_GROUP = 150     # cap per (cohort, diagnosis) group - same as the cohort probe
N_OT_PERMUTATIONS = 20
SEED = 0
LABEL_MAP = {0: "CN", 1: "AD"}
FEATURE_FILES = {
    "3dino": C.FEATURES / "3dino" / "feats.csv",
    "brainiac": C.FEATURES / "brainiac" / "mean.csv",
    "sammed": C.FEATURES / "sammed" / "mean.csv",
    "medicalsam2": C.FEATURES / "medicalsam2" / "mean.csv",
    "curia2_cls": C.FEATURES / "curia2_cls" / "mean.csv",
    "nnfoundation_vit": C.FEATURES / "nnfoundation_vit" / "mean.csv",
}
FEATURE_CONDITIONS = {"original": "Original", "combat_sex": "ComBat (sex)", "combat_sex_age": "ComBat (sex + age)"}


def feature_file(fm: str, cond: str = "original"):
    if cond == "original":
        return FEATURE_FILES[fm]
    return C.DATA / "features_combat" / cond.removeprefix("combat_") / fm / "mean.csv"


def _sfx(cond: str) -> str:
    return "" if cond == "original" else f"_{cond}"


COHORT_COLORS = dict(zip(C.COHORTS, plt.get_cmap("Dark2").colors))
COHORT_NAMES = {c: ("Nigerian Brain" if c == C.NB else c) for c in C.COHORTS}


def load_features(fm: str, cond: str = "original") -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(feature_file(fm, cond))
    return df, [c for c in df.columns if c.startswith("Feature")]


# ============================================================================ 2-Wasserstein
def wasserstein_ot(X_a, X_b, rng, n_permutations=20):
    """Exact OT cost (squared-Euclidean ground cost, uniform weights) and the mean of its permutation null."""
    import ot

    X = np.concatenate([X_a, X_b])
    n_a = len(X_a)
    M = ot.dist(X, X, metric="sqeuclidean")
    a = np.full(n_a, 1 / n_a)
    b = np.full(len(X) - n_a, 1 / (len(X) - n_a))

    def cost(order):
        return ot.emd2(a, b, M[np.ix_(order[:n_a], order[n_a:])])

    observed = cost(np.arange(len(X)))
    null_mean = np.mean([cost(rng.permutation(len(X))) for _ in range(n_permutations)]) if n_permutations else np.nan
    return observed, null_mean


def group_distance_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """wasserstein_rel between every pair of (cohort, diagnosis) groups (CN groups first, then AD)."""
    rng_sample = np.random.default_rng(SEED)        # identical subsets for every FM (same scans, same order)
    rng_perm = np.random.default_rng(SEED + 1)
    X = {}
    for label in (0, 1):
        for cohort in C.COHORTS:
            rows = df[(df["cohort"] == cohort) & (df["label"] == label)][feature_cols].to_numpy(np.float64)
            if len(rows) > N_PER_GROUP:
                rows = rows[rng_sample.choice(len(rows), size=N_PER_GROUP, replace=False)]
            X[f"{cohort}_{LABEL_MAP[label]}"] = rows
    names = list(X)
    mat = pd.DataFrame(0.0, index=names, columns=names)
    for g1, g2 in itertools.combinations(names, 2):
        cost, null = wasserstein_ot(X[g1], X[g2], rng_perm, N_OT_PERMUTATIONS)
        total_var = X[g1].var(axis=0, ddof=1).sum() + X[g2].var(axis=0, ddof=1).sum()
        mat.loc[g1, g2] = mat.loc[g2, g1] = (cost - null) / total_var
    return mat


def ratio_table(mats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Between-cohort (same diagnosis) distance relative to the CN-vs-AD distance inside the cohorts of the pair."""
    rows = []
    for fm, m in mats.items():
        for subset, cohorts in [("all cohorts", C.COHORTS), ("without NB", [c for c in C.COHORTS if c != C.NB])]:
            r_cn, r_both, pairs = [], [], []
            for a, b in itertools.combinations(cohorts, 2):
                dis = np.mean([m.loc[f"{a}_CN", f"{a}_AD"], m.loc[f"{b}_CN", f"{b}_AD"]])
                coh_cn = m.loc[f"{a}_CN", f"{b}_CN"]
                coh_both = np.mean([coh_cn, m.loc[f"{a}_AD", f"{b}_AD"]])
                r_cn.append(coh_cn / dis)
                r_both.append(coh_both / dis)
                pairs.append({"FM": C.FM_LABELS[fm], "subset": subset, "pair": f"{a}-{b}", "disease_dist": dis,
                              "cohort_dist_CN": coh_cn, "cohort_dist_CN_AD": coh_both,
                              "ratio_CN": coh_cn / dis, "ratio_CN_AD": coh_both / dis})
            rows.extend(pairs)
    pair_df = pd.DataFrame(rows)
    summ = (pair_df.groupby(["FM", "subset"], sort=False)
            .agg(median_ratio_CN=("ratio_CN", "median"), q25_CN=("ratio_CN", lambda v: v.quantile(.25)),
                 q75_CN=("ratio_CN", lambda v: v.quantile(.75)), frac_pairs_ratio_gt1_CN=("ratio_CN", lambda v: (v > 1).mean()),
                 median_ratio_CN_AD=("ratio_CN_AD", "median"), median_disease_dist=("disease_dist", "median"),
                 median_cohort_dist_CN=("cohort_dist_CN", "median"), n_pairs=("pair", "size")).reset_index())
    return pair_df, summ


# ---- heat map (same look as distance_rel.ipynb) ----------------------------
DIST_CMAP = LinearSegmentedColormap.from_list("seq_gray", ["#babebe", "#909a99", "#717879", "#4b4f51", "#0b0b0b"])
BORDER_COLORS = {"CN_CN": "#1b9e77", "AD_AD": "#7570b3", "CN_AD": "#d95f02"}
BORDER_LABELS = {"CN_CN": "CN vs CN", "AD_AD": "AD vs AD", "CN_AD": "CN vs AD"}
BORDER_MARGIN = 0.04


def _staircase(r0, c0, k, m):
    v = [(c0 + m, r0 + 1 + m)]
    for j in range(k - 2):
        v += [(c0 + j + 1 - m, r0 + j + 1 + m), (c0 + j + 1 - m, r0 + j + 2 + m)]
    v += [(c0 + k - 1 - m, r0 + k - 1 + m), (c0 + k - 1 - m, r0 + k - m), (c0 + m, r0 + k - m)]
    return v


def draw_heatmap(ax, matrix, title, vmin, vmax, annot_size=14, tick_size=12, lw=4, legend=True, cbar=True, cbar_ax=None):
    labels = [l.replace("_", "-") for l in matrix.index]
    labels = [l.replace(C.NB, "NigerianBrain") for l in labels]
    classes = [l.split("-")[-1] for l in labels]
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, annot=True, fmt=".2f", mask=mask, cmap=DIST_CMAP, square=True, vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor="white", ax=ax, annot_kws={"size": annot_size},
                cbar=cbar, cbar_ax=cbar_ax, cbar_kws={"label": "Relative $W_2^2$ above permutation null"} if cbar else None)
    norm = (matrix.values - vmin) / (vmax - vmin + 1e-8)
    for t in ax.texts:
        x, y = t.get_position()
        t.set_color("white" if norm[int(y), int(x)] > 0.5 else "black")
    bounds = [0] + [i for i in range(1, len(classes)) if classes[i] != classes[i - 1]] + [len(classes)]
    for bi in range(len(bounds) - 1):
        for bj in range(bi + 1):
            r0, r1, c0, c1 = bounds[bi], bounds[bi + 1], bounds[bj], bounds[bj + 1]
            key = f"{classes[r0]}_{classes[c0]}" if classes[r0] == classes[c0] else "CN_AD"
            col = BORDER_COLORS[key]
            if bi == bj:
                k = r1 - r0
                if k >= 2:
                    ax.add_patch(Polygon(_staircase(r0, c0, k, BORDER_MARGIN), closed=True, fill=False, edgecolor=col,
                                         lw=lw, zorder=10, clip_on=False))
            else:
                m = BORDER_MARGIN
                ax.add_patch(plt.Rectangle((c0 + m, r0 + m), (c1 - c0) - 2 * m, (r1 - r0) - 2 * m, fill=False,
                                           edgecolor=col, lw=lw, zorder=10, clip_on=False))
    if legend:
        ax.legend(handles=[Patch(facecolor="none", edgecolor=c, linewidth=3, label=BORDER_LABELS[k])
                           for k, c in BORDER_COLORS.items()], loc="upper right", frameon=True, framealpha=0.9, fontsize=annot_size - 2)
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_size)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, rotation=0, fontsize=tick_size)
    ax.set_title(title, fontsize=tick_size + 6)


def run_w2(fms: list[str], folder, cond: str = "original"):
    sfx, tag = _sfx(cond), FEATURE_CONDITIONS[cond]
    mats = {}
    for fm in fms:
        cache = folder / "tables" / f"w2_rel_groups_{fm}{sfx}.csv"
        if cache.exists():
            mats[fm] = pd.read_csv(cache, index_col=0)
            print(f"[W2] {fm}: loaded {cache.name}")
            continue
        print(f"[W2] computing {fm} [{tag}]")
        df, cols = load_features(fm, cond)
        mats[fm] = group_distance_matrix(df, cols)
        mats[fm].to_csv(cache)
    vmin = min(0.0, min(m.values.min() for m in mats.values()))
    vmax = max(m.values.max() for m in mats.values())
    for fm, m in mats.items():  # one figure per FM
        fig, ax = plt.subplots(figsize=(12, 10))
        draw_heatmap(ax, m, C.FM_LABELS[fm] + ("" if cond == "original" else f" - {tag}"), vmin, vmax)
        fig.savefig(folder / "figures" / f"w2_heatmap_{fm}{sfx}.png", bbox_inches="tight", dpi=300)
        fig.savefig(folder / "figures" / f"w2_heatmap_{fm}{sfx}.pdf", bbox_inches="tight")
        plt.close(fig)
    n = len(mats)  # compact grid of all FMs (one shared colour bar)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(7.2 * ncol + 1, 6.6 * nrow))
    gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1] * ncol + [0.05], wspace=0.25, hspace=0.35)
    cax = fig.add_subplot(gs[:, ncol])
    for k, (fm, m) in enumerate(mats.items()):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        draw_heatmap(ax, m, C.FM_LABELS[fm] + ("" if cond == "original" else f"\n{tag}"), vmin, vmax, annot_size=7, tick_size=8, lw=2, legend=(k == 0),
                     cbar=(k == 0), cbar_ax=cax if k == 0 else None)
    fig.savefig(folder / "figures" / f"w2_heatmap_grid{sfx}.png", bbox_inches="tight", dpi=300)
    fig.savefig(folder / "figures" / f"w2_heatmap_grid{sfx}.pdf", bbox_inches="tight")
    plt.close(fig)
    pair_df, summ = ratio_table(mats)
    C.save_table(pair_df, folder / "tables", f"w2_cohort_vs_disease_ratio_pairs{sfx}")
    C.save_table(summ, folder / "tables", f"w2_cohort_vs_disease_ratio_summary{sfx}")
    return summ


# ============================================================================ t-SNE
# Palette: validated categorical slots (blue, orange, aqua), ordinal blue ramp, neutral greys (see dataviz skill).
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad4"
GREY_BG, GREY_OTHER = "#cfcdc6", "#8f8e88"
DIAG_COLORS = {"CN": "#2a78d6", "AD": "#eb6834"}
MANU_COLORS = {"Siemens": "#2a78d6", "GE": "#eb6834", "Philips": "#1baf7a", "Other / unknown": GREY_OTHER}
FIELD_COLORS = {"0.3 T": "#86b6ef", "1.5 T": "#2a78d6", "3.0 T": "#104281", "unknown": GREY_OTHER}  # ordinal ramp
# 6 cohorts: the only 6-subset of the validated categorical slots with no hard FAIL under all-pairs checks
# (worst CVD dE 6.9 -> legal only with secondary encoding: legend + spatial clusters).
COHORT_COLORS = dict(zip(C.COHORTS, ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]))
COHORT_TITLES = {"NigerianBrain": "Nigerian Brain"}


def rank_fms(available: list[str]) -> list[str]:
    """FMs ordered by mean external AD-vs-CN AUC over all source->target pairs (mean pooling, original features)."""
    d = C.load_cells("pooling_mean")
    auc = d.groupby("model")["auc_disease_mean"].mean()
    return sorted(available, key=lambda m: -auc[m]), auc


def load_embedding(fm: str, folder, meta, cond: str = "original", perplexity=30, seed=42) -> pd.DataFrame:
    from sklearn.manifold import TSNE

    cache = folder / "tables" / f"tsne_embedding_{fm}{_sfx(cond)}.csv"
    if cache.exists():
        e = pd.read_csv(cache)
    else:
        print(f"[t-SNE] {fm} [{FEATURE_CONDITIONS[cond]}]")
        df, cols = load_features(fm, cond)
        xy = TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(df[cols].to_numpy(np.float32))
        e = df[["id", "cohort", "label"]].copy()
        e["tsne1"], e["tsne2"] = xy[:, 0], xy[:, 1]
        e = e.merge(meta, on=["id", "cohort"], how="left")
        e.to_csv(cache, index=False)
    e["Diagnosis"] = e["label"].map({0: "CN", 1: "AD"})
    e["Cohort"] = e["cohort"].map(COHORT_TITLES).fillna(e["cohort"])
    e["Manufacturer"] = e["scanner_manufacturer"].where(e["scanner_manufacturer"].isin(["Siemens", "GE", "Philips"]),
                                                        "Other / unknown")
    e["Field"] = e["field_strength_tesla"].round(1).map(lambda v: "unknown" if pd.isna(v) else f"{v:.1f} T")
    return e


def _draw(ax, e, key, colors, s, order=None, alpha=0.75):
    """Points in random order (no class is systematically drawn on top)."""
    e = e.sample(frac=1.0, random_state=0)
    for k, col in colors.items():
        g = e[e[key] == k]
        ax.scatter(g.tsne1, g.tsne2, s=s, c=col, alpha=alpha, lw=0, rasterized=True)


def _frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GRID)
        sp.set_linewidth(1.0)


def _legend(ax, handles, title):
    ax.axis("off")
    ax.legend(handles=handles, title=title, loc="center left", frameon=False, fontsize=11, title_fontsize=11.5,
              labelcolor=INK2, handletextpad=0.4, borderaxespad=0)
    ax.get_legend().get_title().set_color(INK)
    ax.get_legend()._legend_box.align = 'left'


def _dots(colors, names=None):
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker="o", ls="", ms=8, mfc=c, mec="none", label=(names or {}).get(k, k))
            for k, c in colors.items()]


def tsne_figure(fms: list[str], emb: dict, auc: pd.Series, path, cond: str = "original"):
    """One panel per attribute and FM, colour = category: diagnosis | cohort | scanner manufacturer | field strength."""
    from matplotlib.lines import Line2D
    F = len(fms)
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "text.color": INK})
    fig = plt.figure(figsize=(5.4 * F + 2.4, 14.2), facecolor="white")
    gs = fig.add_gridspec(4, F + 1, width_ratios=[1] * F + [0.38], hspace=0.07, wspace=0.05,
                          left=0.05, right=0.985, top=0.925, bottom=0.03)
    cohort_names = [COHORT_TITLES.get(c, c) for c in C.COHORTS]
    rows = [("Diagnosis", "Diagnosis", DIAG_COLORS, "Diagnosis"),
            ("Cohort", "Cohort", dict(zip(cohort_names, COHORT_COLORS.values())), "Cohort"),
            ("Manufacturer", "Manufacturer", MANU_COLORS, "Scanner\nmanufacturer"),
            ("Field", "Field", FIELD_COLORS, "Field strength")]
    for f, fm in enumerate(fms):
        e = emb[fm]
        title = C.FM_LABELS[fm]
        sub = f"mean external AD-vs-CN AUC {auc[fm]:.2f}" if cond == "original" else FEATURE_CONDITIONS[cond]
        ax0 = None
        for r, (_, key, colors, _) in enumerate(rows):
            ax = fig.add_subplot(gs[r, f])
            ax0 = ax0 or ax
            if key == "Cohort":  # largest cohort first so the small ones stay visible on top
                colors = {k: colors[k] for k in e["Cohort"].value_counts().index}
            _draw(ax, e, key, colors, 6)
            _frame(ax)
        ax0.text(0.5, 1.12, title, transform=ax0.transAxes, ha="center", va="bottom", fontsize=15, fontweight="bold")
        ax0.text(0.5, 1.025, sub, transform=ax0.transAxes, ha="center", va="bottom", fontsize=10.5, color=INK2)
    for r, (lab, key, colors, ltitle) in enumerate(rows):
        _legend(fig.add_subplot(gs[r, -1]), _dots(colors), ltitle)
        bb = gs[r, 0].get_position(fig)
        fig.text(0.016, (bb.y0 + bb.y1) / 2, {"Field": "Field strength (T)", "Manufacturer": "Scanner manufacturer"}.get(lab, lab),
                 rotation=90, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.05, 0.005, "t-SNE of the mean-pooled features of all 3,095 scans (axes are arbitrary and omitted).",
             fontsize=9, color=INK2, va="bottom")
    fig.savefig(f"{path}.png", dpi=300)
    fig.savefig(f"{path}.pdf", dpi=200)
    plt.close(fig)


def run_tsne(fms: list[str], folder, n_main: int = 2, per_supp: int = 2, conds=("original",)):
    """Original features: main-text figure = the n_main FMs with the best mean external AUC, the rest go to
    supplementary figures. ComBat features: all FMs (same order), per_supp per supplementary figure."""
    meta = pd.read_csv(C.META_CSV, usecols=["id", "cohort", "scanner_manufacturer", "field_strength_tesla"])
    ranked, auc = rank_fms(fms)
    print("[t-SNE] FM ranking by mean external AUC:", ", ".join(f"{C.FM_LABELS[m]} {auc[m]:.3f}" for m in ranked))
    for cond in conds:
        emb = {m: load_embedding(m, folder, meta, cond) for m in ranked}
        figs = folder / "figures"
        if cond == "original":
            tsne_figure(ranked[:n_main], emb, auc, figs / "tsne_main")
            rest, name = ranked[n_main:], "tsne_supplement"
        else:
            rest, name = ranked, f"tsne{_sfx(cond)}"
        for k in range(0, len(rest), per_supp):
            tsne_figure(rest[k:k + per_supp], emb, auc, figs / f"{name}_{k // per_supp + 1}", cond)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fms", nargs="+", default=C.FMS, choices=C.FMS)
    ap.add_argument("--skip-tsne", action="store_true")
    ap.add_argument("--skip-w2", action="store_true")
    ap.add_argument("--skip-combat", action="store_true", help="only the original features")
    ap.add_argument("--n-main", type=int, default=2, help="FMs shown in the main-text t-SNE figure (best external AUC)")
    a = ap.parse_args()
    folder = C.outdir("06_feature_space")
    (folder / "tables").mkdir(exist_ok=True)
    (folder / "figures").mkdir(exist_ok=True)
    conds = ["original"] + ([] if a.skip_combat else ["combat_sex", "combat_sex_age"])
    fms = [f for f in a.fms if FEATURE_FILES[f].exists()]
    if len(fms) < len(a.fms):
        print("missing feature files for:", [f for f in a.fms if f not in fms])
    for cond in conds:
        cfms = [f for f in fms if feature_file(f, cond).exists()]
        if not a.skip_w2:
            s = run_w2(cfms, folder, cond)
            pd.set_option("display.width", 200)
            print(f"[{FEATURE_CONDITIONS[cond]}]\n", s.round(3).to_string(index=False))
    if not a.skip_tsne:
        run_tsne(fms, folder, a.n_main, conds=conds)
