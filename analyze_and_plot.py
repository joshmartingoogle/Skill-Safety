#!/usr/bin/env python3
"""
Psychometric, Statistical, and Visual Artefact Generator for Skill Safety Index
-------------------------------------------------------------------------------
For each taxonomy (Stanford TaskNet, O*NET Tasks, O*NET Skills):
  1. Summary statistics overall and by industry (using certainty-weighted Skill Safety Index scores)
  2. Cronbach's alpha for the 6-item index (N1..N5, S), plus full leave-one-out alpha (alpha if item deleted)
     and item-rest correlations to diagnose any item reducing scale reliability
  3. Spot verification audit comparing primary pass vs. independent verification pass on random subsamples
  4. Overall density distribution plots (individual per taxonomy + comparative overlay)
  5. Paneled density distribution plots by industry (2x3 panels across the 6 industries per taxonomy)
"""

import json
import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/usr/local/google/home/joshuabmartin/Skills Project"
OUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DIMS = ["N1", "N2", "N3", "N4", "N5", "S"]
DIM_LABELS = {
    "N1": "N1: Fallback Safety",
    "N2": "N2: Live Performance Safety",
    "N3": "N3: No-AI Rules Safety",
    "N4": "N4: Checking Safety",
    "N5": "N5: Foundation Safety",
    "S":  "S: Stakes Safety",
}

TAXONOMIES = ["Stanford TaskNet", "O*NET Tasks", "O*NET Skills"]
TAX_SLUGS = {
    "Stanford TaskNet": "tasknet",
    "O*NET Tasks": "onet_tasks",
    "O*NET Skills": "onet_skills",
}
TAX_COLORS = {
    "Stanford TaskNet": "#1f4e78",
    "O*NET Tasks": "#2e75b6",
    "O*NET Skills": "#c55a11",
}

INDUSTRY_ORDER = [
    "Healthcare: Clinical Medicine",
    "Legal & Professional Services",
    "Educational Services",
    "Healthcare: Personal & Home Care",
    "Information & Technology",
    "Hospitality & Food Services",
]

INDUSTRY_SHORT = {
    "Healthcare: Clinical Medicine": "Healthcare (Clinical / EM)",
    "Legal & Professional Services": "Legal & Prof. Services",
    "Educational Services": "Educational Services",
    "Healthcare: Personal & Home Care": "Healthcare (Personal Care)",
    "Information & Technology": "Information & Technology",
    "Hospitality & Food Services": "Hospitality & Food Services",
}

INDUSTRY_COLORS = {
    "Healthcare: Clinical Medicine": "#b22222",
    "Legal & Professional Services": "#800080",
    "Educational Services": "#d95f02",
    "Healthcare: Personal & Home Care": "#e6ab02",
    "Information & Technology": "#1b9e77",
    "Hospitality & Food Services": "#2b5c8f",
}


def cronbach_alpha(matrix: np.ndarray) -> float:
    """Computes Cronbach's alpha across columns (items) for rows (observations)."""
    X = np.asarray(matrix, dtype=float)
    k = X.shape[1]
    if k <= 1 or X.shape[0] <= 1:
        return np.nan
    item_vars = np.var(X, axis=0, ddof=1)
    total_scores = np.sum(X, axis=1)
    total_var = np.var(total_scores, axis=0, ddof=1)
    if total_var <= 1e-12:
        return np.nan
    return float((k / (k - 1.0)) * (1.0 - np.sum(item_vars) / total_var))


def compute_reliability_table(df_tax: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes overall Cronbach's alpha (both on certainty-weighted item values and raw item scores)
    and leave-one-out Cronbach's alpha + item-rest correlation for each of N1..N5, S.
    """
    # Build certainty-weighted item matrix: w_{i,k} * x_{i,k} where w_{i,k} = 6 * c_{i,k} / sum_j c_{i,j}
    raw_cols = [f"{d}_score" for d in DIMS]
    cert_cols = [f"{d}_certainty" for d in DIMS]
    X_raw = df_tax[raw_cols].to_numpy(dtype=float)
    C_mat = df_tax[cert_cols].to_numpy(dtype=float)
    C_norm = 6.0 * C_mat / np.sum(C_mat, axis=1, keepdims=True)
    X_cw = X_raw * C_norm

    alpha_cw = cronbach_alpha(X_cw)
    alpha_raw = cronbach_alpha(X_raw)

    loo_rows = []
    for idx, d in enumerate(DIMS):
        keep_idx = [j for j in range(len(DIMS)) if j != idx]
        loo_cw = cronbach_alpha(X_cw[:, keep_idx])
        loo_raw = cronbach_alpha(X_raw[:, keep_idx])

        # Item-rest correlation on certainty-weighted matrix
        item_vec = X_cw[:, idx]
        rest_vec = np.sum(X_cw[:, keep_idx], axis=1)
        if np.std(item_vec) > 1e-9 and np.std(rest_vec) > 1e-9:
            r_it_rest = float(np.corrcoef(item_vec, rest_vec)[0, 1])
        else:
            r_it_rest = 0.0

        loo_rows.append({
            "item": d,
            "label": DIM_LABELS[d],
            "item_mean_cw": round(float(np.mean(X_cw[:, idx])), 3),
            "item_sd_cw": round(float(np.std(X_cw[:, idx], ddof=1)), 3),
            "item_mean_raw": round(float(np.mean(X_raw[:, idx])), 3),
            "mean_certainty": round(float(np.mean(C_mat[:, idx])), 3),
            "item_rest_corr": round(r_it_rest, 3),
            "alpha_if_deleted_cw": round(loo_cw, 4),
            "delta_alpha_cw": round(loo_cw - alpha_cw, 4),
            "alpha_if_deleted_raw": round(loo_raw, 4),
            "delta_alpha_raw": round(loo_raw - alpha_raw, 4),
            "reduces_reliability": bool(loo_cw > alpha_cw),
        })

    return {
        "n_items": int(len(df_tax)),
        "alpha_cw": round(alpha_cw, 4),
        "alpha_raw": round(alpha_raw, 4),
        "below_0_7_threshold": bool(alpha_cw < 0.70),
        "leave_one_out": loo_rows,
    }


def compute_summary_stats(df_tax: pd.DataFrame) -> pd.DataFrame:
    """Computes summary statistics of certainty-weighted Skill Safety Index overall and by industry."""
    rows = []

    def summarize_group(grp: pd.DataFrame, label: str) -> Dict[str, Any]:
        vals = grp["safety_index_cw"].to_numpy(dtype=float)
        norms = grp["safety_index_cw_norm"].to_numpy(dtype=float)
        certs = grp["mean_certainty"].to_numpy(dtype=float)
        gates = grp["gate_ai_assistable"].to_numpy(dtype=bool)
        learn = grp["learning_flag"].to_numpy(dtype=bool)
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        return {
            "Group": label,
            "N": int(len(grp)),
            "AI_Assistable_Pct": round(100.0 * float(np.mean(gates)), 1),
            "Mean_CW": round(float(np.mean(vals)), 3),
            "SD_CW": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 3),
            "Median_CW": round(float(q50), 3),
            "IQR_CW": round(float(q75 - q25), 3),
            "P25_CW": round(float(q25), 3),
            "P75_CW": round(float(q75), 3),
            "Min_CW": round(float(np.min(vals)), 3),
            "Max_CW": round(float(np.max(vals)), 3),
            "Mean_Norm_0_1": round(float(np.mean(norms)), 3),
            "Mean_Certainty": round(float(np.mean(certs)), 3),
            "High_Safety_Share_Pct": round(100.0 * float(np.mean(vals >= 2.0)), 1),
            "Low_Safety_Share_Pct": round(100.0 * float(np.mean(vals < 1.5)), 1),
            "Learner_Bypass_Pct": round(100.0 * float(np.mean(learn)), 1),
        }

    rows.append(summarize_group(df_tax, "Overall (All Industries)"))
    for ind in INDUSTRY_ORDER:
        sub = df_tax[df_tax["industry"] == ind]
        if len(sub) > 0:
            rows.append(summarize_group(sub, INDUSTRY_SHORT[ind]))

    # Also compute AI-assistable gated subset overall
    gated = df_tax[df_tax["gate_ai_assistable"] == True]
    if len(gated) > 0:
        rows.append(summarize_group(gated, "Overall (AI-Assistable Gate = True)"))

    return pd.DataFrame(rows)


def plot_overall_density(df_tax: pd.DataFrame, tax_name: str, slug: str, color: str):
    """Produces an overall kernel density + histogram distribution plot for a single taxonomy."""
    vals = df_tax["safety_index_cw"].to_numpy(dtype=float)
    gated_vals = df_tax[df_tax["gate_ai_assistable"] == True]["safety_index_cw"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=300)
    x_grid = np.linspace(0.0, 3.0, 400)

    # Histogram + KDE for all items
    ax.hist(vals, bins=18, range=(0.0, 3.0), density=True, alpha=0.25, color=color, edgecolor="white", label=f"All Items (N={len(vals)})")
    if len(vals) > 2 and np.std(vals) > 1e-6:
        kde = stats.gaussian_kde(vals, bw_method=0.28)
        y_kde = kde(x_grid)
        ax.plot(x_grid, y_kde, color=color, lw=2.4, label=f"All Items KDE (Mean={np.mean(vals):.2f}, SD={np.std(vals, ddof=1):.2f})")
        ax.fill_between(x_grid, 0, y_kde, color=color, alpha=0.12)

    # KDE for AI-assistable gated subset if different from all items
    if len(gated_vals) > 2 and len(gated_vals) < len(vals) and np.std(gated_vals) > 1e-6:
        kde_g = stats.gaussian_kde(gated_vals, bw_method=0.28)
        ax.plot(x_grid, kde_g(x_grid), color="#333333", lw=1.8, ls="--",
                label=f"AI-Assistable Only (N={len(gated_vals)}, Mean={np.mean(gated_vals):.2f})")

    mean_v = float(np.mean(vals))
    med_v = float(np.median(vals))
    ax.axvline(mean_v, color=color, ls="-", lw=1.6, alpha=0.9)
    ax.axvline(med_v, color="#444444", ls=":", lw=1.5, alpha=0.85, label=f"Median = {med_v:.2f}")

    # Annotate safety zones
    ax.axvspan(0.0, 1.5, color="#f8d7da", alpha=0.18, zorder=0)
    ax.axvspan(2.0, 3.0, color="#d4edda", alpha=0.18, zorder=0)
    ax.set_ylim(0.0, ax.get_ylim()[1] * 1.18)
    ax.text(0.10, ax.get_ylim()[1] * 0.90, "Low Skill Safety\n(High Need / Stakes)", fontsize=7.8, color="#842029", fontweight="bold")
    ax.text(2.22, ax.get_ylim()[1] * 0.90, "High Skill Safety\n(Assist for Good)", fontsize=7.8, color="#0f5132", fontweight="bold")

    ax.set_xlim(0.0, 3.0)
    ax.set_xlabel("Certainty-Weighted Skill Safety Index (0 = Lowest Safety, 3 = Highest Safety)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Density", fontsize=9.5, fontweight="bold")
    ax.set_title(f"{tax_name}: Overall Skill Safety Index Distribution", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper center", frameon=True, facecolor="white", framealpha=0.92, fontsize=7.6)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"density_overall_{slug}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"density_overall_{slug}.png"), dpi=300)
    plt.close(fig)


def plot_combined_overall_density(df_all: pd.DataFrame):
    """Produces a 3-panel side-by-side overall density figure across all 3 taxonomies."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=300, sharey=False)
    x_grid = np.linspace(0.0, 3.0, 400)

    for idx, tax_name in enumerate(TAXONOMIES):
        ax = axes[idx]
        sub = df_all[df_all["taxonomy"] == tax_name]
        vals = sub["safety_index_cw"].to_numpy(dtype=float)
        color = TAX_COLORS[tax_name]

        ax.hist(vals, bins=16, range=(0.0, 3.0), density=True, alpha=0.25, color=color, edgecolor="white")
        if len(vals) > 2 and np.std(vals) > 1e-6:
            kde = stats.gaussian_kde(vals, bw_method=0.28)
            y_kde = kde(x_grid)
            ax.plot(x_grid, y_kde, color=color, lw=2.3, label=f"Mean={np.mean(vals):.2f} (SD={np.std(vals, ddof=1):.2f})")
            ax.fill_between(x_grid, 0, y_kde, color=color, alpha=0.14)

        mean_v = float(np.mean(vals))
        med_v = float(np.median(vals))
        ax.axvline(mean_v, color=color, ls="-", lw=1.6)
        ax.axvline(med_v, color="#333333", ls="--", lw=1.4, label=f"Median = {med_v:.2f}")
        ax.axvspan(0.0, 1.5, color="#f8d7da", alpha=0.16, zorder=0)
        ax.axvspan(2.0, 3.0, color="#d4edda", alpha=0.16, zorder=0)

        ax.set_xlim(0.0, 3.0)
        ax.set_title(f"({chr(65+idx)}) {tax_name} (N={len(vals)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Certainty-Weighted Safety Index (0–3)", fontsize=8.8)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor="white", framealpha=0.9)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "density_overall_all_taxonomies.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "density_overall_all_taxonomies.png"), dpi=300)
    plt.close(fig)


def plot_paneled_industry_density(df_tax: pd.DataFrame, tax_name: str, slug: str):
    """Produces a 2x3 paneled kernel density + histogram plot by industry for a given taxonomy."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), dpi=300, sharex=True, sharey=False)
    axes_flat = axes.flatten()
    x_grid = np.linspace(0.0, 3.0, 300)
    overall_mean = float(df_tax["safety_index_cw"].mean())

    for idx, ind in enumerate(INDUSTRY_ORDER):
        ax = axes_flat[idx]
        sub = df_tax[df_tax["industry"] == ind]
        vals = sub["safety_index_cw"].to_numpy(dtype=float)
        color = INDUSTRY_COLORS[ind]
        short_lbl = INDUSTRY_SHORT[ind]

        ax.hist(vals, bins=12, range=(0.0, 3.0), density=True, alpha=0.28, color=color, edgecolor="white")
        if len(vals) > 2 and np.std(vals) > 1e-6:
            kde = stats.gaussian_kde(vals, bw_method=0.34)
            y_kde = kde(x_grid)
            ax.plot(x_grid, y_kde, color=color, lw=2.2)
            ax.fill_between(x_grid, 0, y_kde, color=color, alpha=0.15)

        mean_v = float(np.mean(vals)) if len(vals) > 0 else np.nan
        sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        med_v = float(np.median(vals)) if len(vals) > 0 else np.nan

        ax.axvline(mean_v, color=color, ls="-", lw=1.7, label=f"Ind. Mean: {mean_v:.2f} (SD: {sd_v:.2f})")
        ax.axvline(med_v, color="#222222", ls=":", lw=1.4, label=f"Ind. Median: {med_v:.2f}")
        ax.axvline(overall_mean, color="#777777", ls="--", lw=1.1, alpha=0.8, label=f"Tax. Mean: {overall_mean:.2f}")

        ax.set_xlim(0.0, 3.0)
        ax.set_title(f"{short_lbl} (N={len(vals)})", fontsize=9.2, fontweight="bold")
        if idx >= 3:
            ax.set_xlabel("Certainty-Weighted Safety Index (0–3)", fontsize=8.5)
        if idx % 3 == 0:
            ax.set_ylabel("Density", fontsize=8.5, fontweight="bold")
        ax.legend(loc="upper left" if mean_v > 1.6 else "upper right", fontsize=7, frameon=True, facecolor="white", framealpha=0.88)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{tax_name}: Certainty-Weighted Skill Safety Density by Industry", fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, f"density_by_industry_{slug}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"density_by_industry_{slug}.png"), dpi=300)
    plt.close(fig)


def analyze_spot_verification(df_main: pd.DataFrame, df_spot: pd.DataFrame) -> Dict[str, Any]:
    """Compares primary scoring pass vs. independent spot-verification pass on random subsample."""
    merged = df_spot.merge(df_main, on="item_id", suffixes=("_spot", "_main"))
    out: Dict[str, Any] = {"overall": {}, "by_taxonomy": {}, "examples": []}

    def calc_metrics(sub: pd.DataFrame) -> Dict[str, Any]:
        cw_m = sub["safety_index_cw_main"].to_numpy(dtype=float)
        cw_s = sub["safety_index_cw_spot"].to_numpy(dtype=float)
        r_cw = float(np.corrcoef(cw_m, cw_s)[0, 1]) if len(sub) > 2 else np.nan
        mae_cw = float(np.mean(np.abs(cw_m - cw_s)))
        gate_agree = float(np.mean(sub["gate_ai_assistable_main"] == sub["gate_ai_assistable_spot"]))

        exact_agrees = []
        adj_agrees = []
        for d in DIMS:
            m_d = sub[f"{d}_score_main"].to_numpy(dtype=int)
            s_d = sub[f"{d}_score_spot"].to_numpy(dtype=int)
            exact_agrees.append(float(np.mean(m_d == s_d)))
            adj_agrees.append(float(np.mean(np.abs(m_d - s_d) <= 1)))

        return {
            "n_subsample": int(len(sub)),
            "index_pearson_r": round(r_cw, 4),
            "index_mae": round(mae_cw, 4),
            "gate_agreement_pct": round(100.0 * gate_agree, 1),
            "mean_dim_exact_agreement_pct": round(100.0 * float(np.mean(exact_agrees)), 1),
            "mean_dim_within_1_agreement_pct": round(100.0 * float(np.mean(adj_agrees)), 1),
            "dim_exact_agreement_pct": {d: round(100.0 * exact_agrees[i], 1) for i, d in enumerate(DIMS)},
        }

    out["overall"] = calc_metrics(merged)
    for tax in TAXONOMIES:
        sub_t = merged[merged["taxonomy_main"] == tax]
        if len(sub_t) > 0:
            out["by_taxonomy"][tax] = calc_metrics(sub_t)
            for _, r in sub_t.head(4).iterrows():
                out["examples"].append({
                    "taxonomy": tax,
                    "item_id": r["item_id"],
                    "occupation": r["occupation_main"],
                    "industry": r["industry_main"],
                    "statement": str(r["statement_main"])[:110] + ("..." if len(str(r["statement_main"])) > 110 else ""),
                    "main_cw": round(float(r["safety_index_cw_main"]), 2),
                    "spot_cw": round(float(r["safety_index_cw_spot"]), 2),
                    "main_cert": round(float(r["mean_certainty_main"]), 2),
                    "dims_main": [int(r[f"{d}_score_main"]) for d in DIMS],
                    "dims_spot": [int(r[f"{d}_score_spot"]) for d in DIMS],
                })

    return out


def main():
    df_all = pd.read_csv(os.path.join(OUT_DIR, "scored_items_all_taxonomies.csv"))
    df_spot = pd.read_csv(os.path.join(OUT_DIR, "spot_verification_subsample.csv"))

    all_artifacts_summary: Dict[str, Any] = {
        "taxonomies": {},
        "spot_verification": {},
    }

    plot_combined_overall_density(df_all)

    for tax_name in TAXONOMIES:
        slug = TAX_SLUGS[tax_name]
        color = TAX_COLORS[tax_name]
        df_tax = df_all[df_all["taxonomy"] == tax_name].copy()

        # 1. Summary statistics overall & by industry
        sum_df = compute_summary_stats(df_tax)
        sum_df.to_csv(os.path.join(OUT_DIR, f"summary_stats_{slug}.csv"), index=False)

        # 2. Cronbach's alpha + leave-one-out
        rel_info = compute_reliability_table(df_tax)
        pd.DataFrame(rel_info["leave_one_out"]).to_csv(os.path.join(OUT_DIR, f"cronbach_loo_{slug}.csv"), index=False)

        # 3. Plots (overall density + paneled density by industry)
        plot_overall_density(df_tax, tax_name, slug, color)
        plot_paneled_industry_density(df_tax, tax_name, slug)

        all_artifacts_summary["taxonomies"][tax_name] = {
            "summary_stats": sum_df.to_dict(orient="records"),
            "reliability": rel_info,
        }

    # 4. Spot verification
    spot_res = analyze_spot_verification(df_all, df_spot)
    all_artifacts_summary["spot_verification"] = spot_res

    with open(os.path.join(OUT_DIR, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_artifacts_summary, f, indent=2)

    print("Successfully generated all tables, reliability diagnostics, and density plots!")
    for tax_name in TAXONOMIES:
        rel = all_artifacts_summary["taxonomies"][tax_name]["reliability"]
        ov = all_artifacts_summary["taxonomies"][tax_name]["summary_stats"][0]
        print(f"=== {tax_name} (N={ov['N']}) ===")
        print(f"  Overall CW Safety Index: Mean={ov['Mean_CW']:.3f}, SD={ov['SD_CW']:.3f}, Median={ov['Median_CW']:.3f}, IQR={ov['IQR_CW']:.3f}, Certainty={ov['Mean_Certainty']:.3f}")
        print(f"  Cronbach's Alpha (CW): {rel['alpha_cw']:.4f} (Raw: {rel['alpha_raw']:.4f}) | Below 0.7: {rel['below_0_7_threshold']}")
        for r in rel["leave_one_out"]:
            print(f"    - LOO without {r['item']} ({r['label']}): alpha_cw={r['alpha_if_deleted_cw']:.4f} (delta={r['delta_alpha_cw']:+.4f}, r_rest={r['item_rest_corr']:.3f})")


if __name__ == "__main__":
    main()
