"""
eda_plots.py
============
Generates univariate, bivariate, and multivariate EDA plots for the
Smart Lender dataset. Saves them as PNG images to static/images/eda/.

Run standalone:
    python Flask/eda_plots.py

Or import and call generate_all_plots().
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Make the shared preprocessing module importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Flask"))

import preprocessing as pp  # noqa: E402

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

DATA_PATH = PROJECT_ROOT / "Dataset" / "loan_prediction.csv"
OUT_DIR = PROJECT_ROOT / "Flask" / "static" / "images" / "eda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


# ---------------------------------------------------------------------------
# 1. Univariate Analysis
# ---------------------------------------------------------------------------
def plot_univariate(df: pd.DataFrame):
    """Generate univariate plots: histograms for numeric, bar charts for categorical."""
    _ensure_dir(OUT_DIR)

    # --- Numeric features ---
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                    "Loan_Amount_Term", "Credit_History"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        if col in df.columns:
            ax = axes[i]
            data = df[col].dropna()
            if col == "Credit_History":
                # Credit_History is discrete (0/1)
                counts = data.value_counts().sort_index()
                ax.bar(counts.index.astype(str), counts.values,
                       color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=1.5)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                for j, v in enumerate(counts.values):
                    ax.text(j, v + 5, str(v), ha="center", fontweight="bold")
            else:
                ax.hist(data, bins=25, color="#3498db", edgecolor="white",
                        linewidth=1.2, alpha=0.85)
                ax.axvline(data.median(), color="#e74c3c", linestyle="--",
                           linewidth=2, label=f"Median={data.median():.0f}")
                ax.axvline(data.mean(), color="#2ecc71", linestyle="--",
                           linewidth=2, label=f"Mean={data.mean():.0f}")
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.legend(fontsize=10)
    # Hide unused subplot
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])
    fig.suptitle("Univariate Analysis — Numeric Features", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "univariate_numeric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Categorical features ---
    cat_cols = ["Gender", "Married", "Dependents", "Education",
                "Self_Employed", "Property_Area", "Loan_Status"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()
    colors = sns.color_palette("viridis", 8)
    for i, col in enumerate(cat_cols):
        if col in df.columns:
            ax = axes[i]
            counts = df[col].value_counts()
            bars = ax.bar(counts.index.astype(str), counts.values,
                          color=colors[:len(counts)], edgecolor="white", linewidth=1.5)
            ax.set_title(f"Distribution of {col}", fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 2,
                        f"{int(height)}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    # Hide unused subplots
    for i in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[i])
    fig.suptitle("Univariate Analysis — Categorical Features", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "univariate_categorical.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Target distribution (Loan_Status) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["Loan_Status"].value_counts()
    colors_target = ["#2ecc71" if k == "Y" else "#e74c3c" for k in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=colors_target, startangle=90, explode=(0.05, 0.05),
        shadow=False, textprops={"fontsize": 14, "fontweight": "bold"}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Target Variable: Loan_Status (Approved vs Rejected)",
                 fontsize=14, fontweight="bold", pad=20)
    fig.savefig(OUT_DIR / "univariate_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Bivariate Analysis
# ---------------------------------------------------------------------------
def plot_bivariate(df: pd.DataFrame):
    """Generate bivariate plots: correlation heatmap, cross-tabulations."""
    _ensure_dir(OUT_DIR)

    # --- Correlation heatmap (numeric features only) ---
    # Select numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Loan_ID"], errors="ignore")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                fmt=".2f", linewidths=1, square=True, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title("Bivariate Analysis — Correlation Matrix (Numeric Features)",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "bivariate_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Loan_Status vs categorical features (grouped bar charts) ---
    cat_features = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area", "Credit_History"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()
    for i, col in enumerate(cat_features):
        if col in df.columns:
            ax = axes[i]
            ct = pd.crosstab(df[col].astype(str), df["Loan_Status"])
            ct.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"],
                    edgecolor="white", linewidth=1.2, legend=False)
            ax.set_title(f"Loan_Status vs {col}", fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            # Add legend only to first plot
            if i == 0:
                ax.legend(["Rejected (N)", "Approved (Y)"], fontsize=10)
    for i in range(len(cat_features), len(axes)):
        fig.delaxes(axes[i])
    fig.suptitle("Bivariate Analysis — Loan_Status vs Categorical Features",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "bivariate_categorical.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Income vs LoanAmount by Loan_Status ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for status, color, marker in [("Y", "#2ecc71", "o"), ("N", "#e74c3c", "s")]:
        subset = df[df["Loan_Status"] == status].dropna(subset=["ApplicantIncome", "LoanAmount"])
        ax.scatter(subset["ApplicantIncome"], subset["LoanAmount"],
                   c=color, label=f"Approved" if status == "Y" else "Rejected",
                   alpha=0.6, edgecolors="white", linewidth=0.5, s=60, marker=marker)
    ax.set_xlabel("Applicant Income")
    ax.set_ylabel("Loan Amount (in thousands)")
    ax.set_title("Applicant Income vs Loan Amount by Loan Status",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "bivariate_income_vs_loan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Multivariate Analysis
# ---------------------------------------------------------------------------
def plot_multivariate(df: pd.DataFrame):
    """Generate multivariate plots: pairplot, box plots, etc."""
    _ensure_dir(OUT_DIR)

    # --- Pairplot of key numeric features ---
    sample_df = df.drop(columns=["Loan_ID"], errors="ignore")
    # Encode Loan_Status for coloring
    plot_df = sample_df.copy()
    plot_df["Loan_Status_Encoded"] = plot_df["Loan_Status"].map({"Y": "Approved", "N": "Rejected"})

    # Numeric features for pairplot
    pair_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    # Add encoded status for hue
    pair_df = plot_df[pair_features + ["Loan_Status_Encoded"]].dropna()

    g = sns.PairGrid(pair_df, hue="Loan_Status_Encoded",
                     palette={"Approved": "#2ecc71", "Rejected": "#e74c3c"},
                     diag_sharey=False, corner=False)
    g.map_upper(sns.scatterplot, alpha=0.6, s=40, edgecolor="white", linewidth=0.5)
    g.map_lower(sns.kdeplot, alpha=0.3, levels=4, fill=True)
    g.map_diag(sns.histplot, alpha=0.7, edgecolor="white", linewidth=0.8, bins=20)
    g.add_legend(title="Loan Status", fontsize=11)
    g.fig.suptitle("Multivariate Analysis — Pairplot of Numeric Features",
                   fontsize=16, fontweight="bold", y=1.02)
    g.fig.savefig(OUT_DIR / "multivariate_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    # --- Box plots grouped by Loan_Status ---
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        df_box = df.dropna(subset=[col, "Loan_Status"])
        bp = ax.boxplot(
            [df_box[df_box["Loan_Status"] == s][col].values for s in ["Y", "N"]],
            labels=["Approved (Y)", "Rejected (N)"],
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "white", "linewidth": 2},
        )
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        ax.set_title(f"{col} by Loan Status", fontweight="bold")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Multivariate Analysis — Box Plots Grouped by Loan Status",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "multivariate_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Violin plots for key distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, col in enumerate(["ApplicantIncome", "LoanAmount"]):
        ax = axes[i]
        df_violin = df.dropna(subset=[col, "Loan_Status"])
        parts = ax.violinplot(
            [df_violin[df_violin["Loan_Status"] == s][col].values for s in ["Y", "N"]],
            positions=[1, 2], showmeans=True, showmedians=True, widths=0.7
        )
        parts["bodies"][0].set_facecolor("#2ecc71")
        parts["bodies"][0].set_alpha(0.7)
        parts["bodies"][1].set_facecolor("#e74c3c")
        parts["bodies"][1].set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Approved (Y)", "Rejected (N)"])
        ax.set_ylabel(col)
        ax.set_title(f"{col} Distribution by Loan Status (Violin Plot)", fontweight="bold")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Multivariate Analysis — Violin Plots",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "multivariate_violin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------
def generate_all_plots():
    """Run all EDA plot generation functions."""
    print("[EDA] Loading data...")
    df = _load_data()
    print(f"[EDA] Loaded dataset: {df.shape}")

    print("[EDA] Generating univariate plots...")
    plot_univariate(df)

    print("[EDA] Generating bivariate plots...")
    plot_bivariate(df)

    print("[EDA] Generating multivariate plots...")
    plot_multivariate(df)

    print(f"[EDA] All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    generate_all_plots()
