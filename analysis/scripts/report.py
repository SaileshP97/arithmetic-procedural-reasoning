"""Tables and figures for the camera-ready analyses, from outputs/per_example.parquet.

C2  intermediate-state verification: correct-prefix length, first error position   (JsBn W1, TMAk)
C8  error type at the first non-faithful step                                       (TMAk)
C5  intermediate-value statistics of the reference executor vs. FAA                (JsBn)

Writes CSV + LaTeX tables to outputs/ and PDF figures to figures/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = os.path.join(HERE, "outputs"), os.path.join(HERE, "figures")
PAPER_PARQUET = "/Volumes/Crucial/openreview_attachments/extracted/unified_results.parquet"

ABBR = {
    "DeepSeek-R1-Distill-Qwen-1.5B": "DS-1.5", "Qwen3-4B-Thinking-2507": "QW-4", "Olmo-3-7B-Think": "OL-7",
    "Mistral-7B-Instruct-v0.3": "MI-7", "DeepSeek-R1-Distill-Qwen-7B": "DS-7", "openthinker2": "OT-7",
    "Ministral-3-14B-Reasoning-2512": "MN-14", "DeepSeek-R1-Distill-Qwen-14B": "DS-14", "Qwen3-14B": "QW-14",
    "Magistral-Small-2509": "MG-24", "sarvam-30B": "SV-30", "Nemotron-3-Nano30B": "NM-30",
    "gpt-oss-120b": "OA-120", "deepseek-v3.2": "DS-685", "kimi-k2.5": "KK-1T",
}
ORDER = list(ABBR.values())
PARTIAL = {"DS-685", "KK-1T"}

# palette (validated: scripts in the dataviz skill, light mode) -------------------------------
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NEUTRAL = "#c9c8c3"
SEQ = ["#86b6ef", "#5598e7", "#256abf", "#104281"]  # ordinal blue ramp, light -> dark

ERR_ORDER = ["arithmetic_error", "wrong_operand", "operand_order_swapped", "wrong_operation", "skipped_step", "trace_ends"]
ERR_LABEL = {
    "all_faithful": "All steps correct", "arithmetic_error": "Arithmetic error", "wrong_operand": "Wrong operand (state retrieval)",
    "operand_order_swapped": "Operand order swapped", "wrong_operation": "Wrong operation", "skipped_step": "Skipped step",
    "trace_ends": "Trace ends early", "no_intermediate_states": "No intermediate states",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "pdf.fonttype": 42, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def save_table(df: pd.DataFrame, name: str, caption: str, fmt="{:.1f}"):
    df.to_csv(os.path.join(OUT, f"{name}.csv"))
    body = df.copy()
    for c in body.columns:
        if pd.api.types.is_float_dtype(body[c]):
            body[c] = body[c].map(lambda v: "--" if pd.isna(v) else fmt.format(v))

    def tx(s):  # LaTeX-safe labels (index labels may already contain math such as $^\dagger$)
        s = str(s)
        if "$" in s:
            return s
        for a, b in [("%", "\\%"), ("_", "\\_"), ("&", "\\&"), ("<", "$<$"), (">", "$>$"), ("|", "$|$"),
                     ("≥", "$\\geq$"), ("²", "$^{2}$"), ("³¹", "$^{31}$"), ("¹³", "$^{13}$"), ("⁴", "$^{4}$"), ("⁷", "$^{7}$")]:
            s = s.replace(a, b)
        return s
    body.columns = [tx(c) for c in body.columns]
    body.index = [tx(i) for i in body.index]
    body.index.name = tx(df.index.name) if df.index.name else None
    if body.columns.name:
        body.columns.name = tx(body.columns.name)
    cap = caption.replace("%", "\\%") if "\\%" not in caption else caption
    tex = body.to_latex(escape=False, caption=cap, label=f"tab:{name}", position="H")
    tex = tex.replace("\\begin{table}[H]", "\\begin{table}[H]\n\\small\\centering")
    if body.shape[1] >= 5:  # wide tables: scale to the text width
        tex = tex.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")
    open(os.path.join(OUT, f"{name}.tex"), "w").write(tex)


def load() -> pd.DataFrame:
    d = pd.read_parquet(os.environ.get("PER_EXAMPLE", os.path.join(OUT, "per_example.parquet")))
    d["M"] = d.model.map(ABBR)
    d["visible"] = d.n_parsed_states > 0
    ev = d.n_eval_steps.clip(lower=1)
    d["faithful_frac"] = d.faithful_prefix / ev
    d["faithful_loose_frac"] = d.faithful_prefix_loose / ev
    for t in ("0.001", "0.01", "0.05"):
        d[f"ref_frac_{t}"] = d[f"ref_prefix_first_{t}"] / ev
    d["ref_frac_last"] = d["ref_prefix_last_0.01"] / ev
    d["all_faithful"] = d.error_type.isin(["all_faithful", "all_correct_until_ref_nonfinite"])
    d["first_error_rel"] = d.first_error_step / d.num_steps
    # paper scoring (OpenReview data attachment), joined by model + config + id
    if os.path.exists(PAPER_PARQUET):
        p = pd.read_parquet(PAPER_PARQUET, columns=["model", "range", "dtype", "look_back", "task", "id", "correct"])
        p["key"] = p.model.str.lower()
        p["look_back"] = p.look_back.str.replace("look_back", "").astype(int)
        d["key"] = d.model.str.lower()
        d = d.merge(p.drop(columns="model").rename(columns={"correct": "correct_paper"}),
                    on=["key", "range", "dtype", "look_back", "task", "id"], how="left")
    else:
        d["correct_paper"] = np.nan
    d["FAA"] = d.correct_paper.fillna(d.correct_raw).astype(bool)
    return d


# ================================================================== C2
def c2(d: pd.DataFrame):
    g = d.groupby("M")
    vis = d[d.visible].groupby("M")
    t = pd.DataFrame({
        "N": g.size(),
        "FAA (%)": g.FAA.mean() * 100,
        "Visible states (%)": g.visible.mean() * 100,
        "State coverage (%)": vis.apply(lambda x: (x.n_parsed_states / x.num_steps).clip(upper=1).mean() * 100),
        "Ref.-matching prefix (%)": vis["ref_frac_0.01"].mean() * 100,
        "Correct prefix (%)": vis.faithful_frac.mean() * 100,
        "All steps correct (%)": vis.all_faithful.mean() * 100,
        "Median first error (rel.)": vis.first_error_rel.median(),
    }).reindex(ORDER)
    t.index = [f"{m}$^\\dagger$" if m in PARTIAL else m for m in t.index]
    save_table(t, "c2_intermediate_state_by_model",
               "Intermediate-state verification against the reference executor. Prefix metrics are computed over "
               "examples with at least one parseable intermediate state. $^\\dagger$Traces available for a subset only.")

    # faithfulness vs final correctness
    v = d[d.visible]
    x = v.groupby("M").apply(lambda s: pd.Series({
        "P(all steps correct | FAA correct) (%)": s[s.FAA].all_faithful.mean() * 100 if s.FAA.any() else np.nan,
        "P(FAA correct | all steps correct) (%)": s[s.all_faithful].FAA.mean() * 100 if s.all_faithful.any() else np.nan,
        "Correct answer, incorrect step (%)": (s.FAA & ~s.all_faithful).mean() * 100,
    })).reindex(ORDER)
    save_table(x, "c2_faithfulness_vs_correctness",
               "Relation between correct step execution and final-answer correctness (examples with visible states).")

    # by procedure length (pooled over models with >=50% visible traces, full-trace models only)
    keep = [m for m in t.index if not m.endswith("$") and t.loc[m, "Visible states (%)"] >= 50]
    pool = v[v.M.isin(keep)]
    bys = pool.groupby("num_steps").agg(
        faithful_prefix=("faithful_frac", "mean"), ref_prefix=("ref_frac_0.01", "mean"),
        all_faithful=("all_faithful", "mean"), FAA=("FAA", "mean"), n=("id", "size")) * [100, 100, 100, 100, 1]
    save_table(bys, "c2_by_steps_pooled", f"Intermediate-state metrics by procedure length, pooled over {', '.join(keep)}.")
    # by look-back (mixed operations only)
    byl = v[(v.M.isin(keep)) & (v.task == "mixed")].groupby("look_back").agg(
        faithful_prefix=("faithful_frac", "mean"), ref_prefix=("ref_frac_0.01", "mean"),
        all_faithful=("all_faithful", "mean"), FAA=("FAA", "mean"), n=("id", "size")) * [100, 100, 100, 100, 1]
    save_table(byl, "c2_by_lookback_pooled", "Intermediate-state metrics by look-back depth (mixed operations).")

    # tolerance sensitivity
    sens = v.groupby("M")[["ref_frac_0.001", "ref_frac_0.01", "ref_frac_0.05", "ref_frac_last", "faithful_frac", "faithful_loose_frac"]].mean().reindex(ORDER) * 100
    sens.columns = ["Ref. 0.1%", "Ref. 1%", "Ref. 5%", "Ref. 1% (last value)", "Correct prefix (0.1%/ulp)", "Correct prefix (1%)"]
    save_table(sens, "c2_tolerance_sensitivity", "Sensitivity of prefix metrics (mean % of steps) to tolerance and to using the last instead of the first value per variable.")

    # common subset: configurations available for all 15 models
    common = d[(d["range"] == "range_0_1") & (d["dtype"] == "float") & (d.look_back == 1) & (d.task.isin(["add", "mixed"]))]
    cs = common[common.visible].groupby("M").agg(N=("id", "size"), faithful_prefix=("faithful_frac", "mean"),
                                                  all_faithful=("all_faithful", "mean"), FAA=("FAA", "mean")).reindex(ORDER)
    cs[["faithful_prefix", "all_faithful", "FAA"]] *= 100
    save_table(cs, "c2_common_subset", "Intermediate-state metrics on the configurations available for all 15 models (range 0--1, float, look-back 1, add and mixed).")

    # figure: small multiples, prefix vs procedure length per model
    models = [m for m in ORDER if m in set(v.M)]
    cols = 5
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 1.45 * rows), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, models):
        s = v[v.M == m].groupby("num_steps")[["faithful_frac", "ref_frac_0.01"]].mean() * 100
        ax.plot(s.index, s["ref_frac_0.01"], color=SERIES[1], lw=1.5, label="Matches reference (1%)")
        ax.plot(s.index, s.faithful_frac, color=SERIES[0], lw=1.5, label="Correct prefix (own values)")
        ax.set_title(f"{m}{'†' if m in PARTIAL else ''}", color=INK, pad=2, fontweight="bold")
        ax.text(0.97, 0.93, f"visible {d[d.M == m].visible.mean() * 100:.0f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=6, color=INK2)
        ax.set_ylim(0, 100)
        ax.set_xticks([5, 50, 95])
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    fig.supylabel("Correct prefix (% of steps)", fontsize=8, color=INK)
    fig.supxlabel("Procedure length (steps)", fontsize=8, color=INK)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.5, w_pad=0.3)
    fig.savefig(os.path.join(FIG, "c2_prefix_vs_steps.pdf"), bbox_inches="tight")
    plt.close(fig)

    # figure: distribution of first error position (relative)
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    e = pool.dropna(subset=["first_error_rel"])
    bins = np.linspace(0, 1, 21)
    ax.hist(e.first_error_rel, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
    ax.set_xlabel("First incorrect step (fraction of procedure)")
    ax.set_ylabel("Examples")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "c2_first_error_position.pdf"), bbox_inches="tight")
    plt.close(fig)
    return t


# ================================================================== C8
def c8(d: pd.DataFrame):
    v = d[d.visible]
    ct = pd.crosstab(v.M, v.error_type, normalize="index").reindex(ORDER).dropna(how="all") * 100
    cols = ["all_faithful"] + [c for c in ERR_ORDER if c in ct.columns]
    ct = ct.reindex(columns=cols, fill_value=0)
    vis = d.groupby("M").visible.mean().reindex(ct.index) * 100
    out = ct.rename(columns=ERR_LABEL)
    out.insert(0, "Visible states (%)", vis)
    out.index = [f"{m}$^\\dagger$" if m in PARTIAL else m for m in out.index]
    save_table(out, "c8_error_types_by_model",
               "Category of the first incorrect step (% of examples with visible intermediate states).")

    errs = v[v.error_type.isin(ERR_ORDER)]
    bys = pd.crosstab(errs.num_steps, errs.error_type, normalize="index").reindex(columns=ERR_ORDER, fill_value=0) * 100
    save_table(bys.rename(columns=ERR_LABEL), "c8_error_types_by_steps", "Error-type composition by procedure length (erroneous traces, all models).")
    byl = errs[errs.task == "mixed"]
    byl = pd.crosstab(byl.look_back, byl.error_type, normalize="index").reindex(columns=ERR_ORDER, fill_value=0) * 100
    save_table(byl.rename(columns=ERR_LABEL), "c8_error_types_by_lookback", "Error-type composition by look-back depth (mixed operations, erroneous traces).")

    # figure: stacked horizontal bars, errors only (share among erroneous visible traces)
    e = pd.crosstab(errs.M, errs.error_type, normalize="index").reindex(columns=ERR_ORDER, fill_value=0).reindex([m for m in ORDER if m in set(errs.M)]) * 100
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    left = np.zeros(len(e))
    y = np.arange(len(e))[::-1]
    for i, c in enumerate(ERR_ORDER):
        ax.barh(y, e[c].values, left=left, color=SERIES[i], edgecolor=SURFACE, linewidth=1.0, height=0.72, label=ERR_LABEL[c])
        left += e[c].values
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}†" if m in PARTIAL else m for m in e.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of erroneous traces (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=2, frameon=False, handlelength=1.0, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "c8_error_types.pdf"), bbox_inches="tight")
    plt.close(fig)


# ================================================================== C5
def c5(d: pd.DataFrame):
    # reference statistics are properties of the dataset: compute on one row per example id
    r = d.drop_duplicates("id")
    r = r[r.procedure_source == "dataset"]
    r["has_nonfinite"] = r.ref_n_nonfinite > 0
    stats = r.groupby(["task", "num_steps"]).agg(
        median_max_log10=("ref_max_log10", "median"), p90_max_log10=("ref_max_log10", lambda s: s.quantile(0.9)),
        median_min_nonzero_log10=("ref_min_nonzero_log10", "median"), nonfinite_pct=("has_nonfinite", "mean"), n=("id", "size"))
    stats["nonfinite_pct"] *= 100
    save_table(stats, "c5_reference_intermediate_stats_by_task_steps",
               "Reference intermediate values: $\\lfloor\\log_{10}|v|\\rfloor$ of the largest and smallest non-zero intermediate value, and % of examples with a non-finite intermediate value.")
    st2 = r.groupby(["range", "dtype"]).agg(median_max_log10=("ref_max_log10", "median"), median_min_nonzero_log10=("ref_min_nonzero_log10", "median"),
                                            nonfinite_pct=("has_nonfinite", "mean"))
    st2["nonfinite_pct"] *= 100
    save_table(st2, "c5_reference_intermediate_stats_by_range_dtype", "Reference intermediate values by input range and data type.")

    # FAA by max-magnitude bin, within procedure-length groups (all 13 full models; finite references only)
    full = d[~d.M.isin(PARTIAL) & (d.ref_n_nonfinite == 0)].copy()
    full["len_group"] = pd.cut(full.num_steps, [0, 15, 45, 75, 95], labels=["5–15", "25–45", "55–75", "85–95"])
    full["mag_bin"] = pd.cut(full.ref_max_log10, [-np.inf, 1, 3, 6, 12, 30, np.inf],
                             labels=["<10²", "10²–10⁴", "10⁴–10⁷", "10⁷–10¹³", "10¹³–10³¹", "≥10³¹"])
    tab = full.pivot_table(index="mag_bin", columns="len_group", values="FAA", aggfunc="mean", observed=False) * 100
    cnt = full.pivot_table(index="mag_bin", columns="len_group", values="FAA", aggfunc="size", observed=False)
    save_table(tab, "c5_faa_by_magnitude_and_length", "FAA (%) by magnitude of the largest reference intermediate value, within procedure-length groups (13 full-trace models).")
    cnt.to_csv(os.path.join(OUT, "c5_faa_by_magnitude_and_length_counts.csv"))

    # non-finite: FAA with and without examples whose reference has a non-finite value
    nf = d.groupby("M").apply(lambda s: pd.Series({
        "FAA all (%)": s.FAA.mean() * 100,
        "FAA finite refs (%)": s[s.ref_n_nonfinite == 0].FAA.mean() * 100,
        "Non-finite refs (%)": (s.ref_n_nonfinite > 0).mean() * 100,
    })).reindex(ORDER)
    save_table(nf, "c5_faa_nonfinite_sensitivity", "FAA with and without examples whose reference execution contains a non-finite intermediate value.")

    # figure
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    x = np.arange(len(tab.index))
    for i, c in enumerate(tab.columns):
        ok = cnt[c].values >= 200  # hide bins with too few examples
        ax.plot(x[ok], tab[c].values[ok], color=SEQ[i], lw=1.5, marker="o", ms=3.5, label=f"{c} steps")
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index, rotation=0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Largest intermediate value |v| (reference)")
    ax.set_ylabel("FAA (%)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "c5_faa_vs_magnitude.pdf"), bbox_inches="tight")
    plt.close(fig)


def coverage(d: pd.DataFrame):
    c = d.groupby("M").agg(examples=("id", "size"), configs=("task", lambda s: 0)).reindex(ORDER)
    c["configs"] = d.groupby("M").apply(lambda s: s[["range", "dtype", "look_back", "task"]].drop_duplicates().shape[0]).reindex(ORDER)
    c["procedure inferred"] = d.groupby("M").apply(lambda s: (s.procedure_source != "dataset").sum()).reindex(ORDER)
    save_table(c, "coverage", "Stored generations used for the intermediate-state analysis.", fmt="{:.0f}")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    d = load()
    coverage(d)
    t = c2(d)
    c8(d)
    c5(d)
    print(t.round(1).to_string())
    print("written to", OUT, "and", FIG)


# ================================================================== main-text figure (two panels)
def fig_main(d: pd.DataFrame):
    """(a) pooled faithful prefix / fully faithful / FAA vs length; (b) first-error composition vs length."""
    full = [m for m in ORDER if m not in PARTIAL]
    vis = d[d.visible & d.M.isin(full)]
    share = d[d.M.isin(full)].groupby("M").visible.mean()
    keep = share[share >= 0.5].index
    v = vis[vis.M.isin(keep)]
    g = v.groupby("num_steps")
    a = pd.DataFrame({"Correct prefix": g.faithful_frac.mean() * 100,
                      "All steps correct": g.all_faithful.mean() * 100,
                      "FAA": g.FAA.mean() * 100})
    errs = v[v.error_type.isin(ERR_ORDER)]
    b = pd.crosstab(errs.num_steps, errs.error_type, normalize="index").reindex(columns=ERR_ORDER, fill_value=0) * 100
    a.to_csv(os.path.join(OUT, "fig_main_panel_a.csv"))
    b.to_csv(os.path.join(OUT, "fig_main_panel_b.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 2.3), gridspec_kw={"width_ratios": [1, 1.05]})
    styles = [(SERIES[0], "-"), (SERIES[1], "-"), (INK2, "--")]
    for (col, (c, ls)) in zip(a.columns, styles):
        ax1.plot(a.index, a[col], color=c, ls=ls, lw=1.5, label=col)
    ax1.set_ylim(0, 100)
    ax1.set_xticks([5, 35, 65, 95])
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("%")
    ax1.set_title("(a) Step-level verification", fontsize=7, pad=2)
    ax1.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.45, -0.36), ncol=1,
               handlelength=1.6, borderaxespad=0.0)
    x = np.arange(len(b.index))
    bottom = np.zeros(len(b))
    for i, c in enumerate(ERR_ORDER):
        ax2.bar(x, b[c].values, bottom=bottom, color=SERIES[i], edgecolor=SURFACE, linewidth=0.6, width=0.8, label=ERR_LABEL[c])
        bottom += b[c].values
    ax2.set_xticks(x[::3])
    ax2.set_xticklabels(b.index[::3])
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Steps")
    ax2.set_title("(b) Type of first error", fontsize=7, pad=2)
    ax2.grid(axis="x", visible=False)
    h, l = ax2.get_legend_handles_labels()
    ax2.legend(h, [s.replace(" (state retrieval)", "") for s in l], loc="upper center", ncol=2, frameon=False,
               fontsize=5.5, handlelength=1.0, columnspacing=0.6, bbox_to_anchor=(0.5, -0.36), borderaxespad=0.0)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, "step_verification_main.pdf"), bbox_inches="tight")
    plt.close(fig)
    return list(keep)


if __name__ == "__main__":
    print("main-figure models:", fig_main(load()))
