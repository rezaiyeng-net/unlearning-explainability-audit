"""
Professional Visualization for Unlearning Experiment Results
Generates publication-ready figures (300 DPI, serif fonts).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import rcParams

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Publication-ready style
rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (14, 10),
})

# Standardize names
DATASET_MAP = {
    'german': 'German Credit', 'German Credit': 'German Credit',
    'adult': 'Adult', 'Adult': 'Adult',
    'heart': 'Heart Disease', 'Heart Disease': 'Heart Disease'
}
MODEL_MAP = {'LR': 'LR', 'RF': 'RF', 'GB': 'GB', 'MLP': 'MLP'}
MODEL_ORDER = ['LR', 'RF', 'GB', 'MLP']
DATASET_ORDER = ['German Credit', 'Adult', 'Heart Disease']
METHOD_COLORS = {'Exact': '#2E86AB', 'SISA': '#A23B72', 'Weighting': '#F18F01'}


# ============================================================
# Data Loading
# ============================================================
def load_all_results() -> pd.DataFrame:
    dfs = []
    for fname in ['part1_results.csv', 'part2_results.csv']:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            df['dataset'] = df['dataset'].map(DATASET_MAP)
            df['model'] = df['model'].map(MODEL_MAP)
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    if 'mia' not in combined.columns:
        combined['mia'] = np.nan
    return combined


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['dataset', 'model', 'forget_size', 'method']).agg(
        acc_mean=('accuracy', 'mean'),
        acc_std=('accuracy', 'std'),
        fid_mean=('fid', 'mean'),
        fid_std=('fid', 'std'),
        mia_mean=('mia', 'mean'),
        mia_std=('mia', 'std'),
    ).reset_index()


def cohens_d(m1, s1, m2, s2):
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    if pooled == 0:
        return 0.0
    return (m1 - m2) / pooled


# ============================================================
# Figure 1: Bar Plots (Accuracy & FID per dataset)
# ============================================================
def figure1_bar_plots(summary: pd.DataFrame, forget_size: int = 5):
    """Comparison of unlearning methods across three datasets and four model architectures."""
    subset = summary[summary['forget_size'] == forget_size]
    
    fig, axes = plt.subplots(len(DATASET_ORDER), 2, figsize=(14, 12))
    
    for i, dataset in enumerate(DATASET_ORDER):
        ds_data = subset[subset['dataset'] == dataset]
        
        # Accuracy subplot
        ax1 = axes[i, 0]
        x = np.arange(len(MODEL_ORDER))
        width = 0.25
        for j, method in enumerate(['Exact', 'SISA', 'Weighting']):
            method_data = ds_data[ds_data['method'] == method].set_index('model')
            method_data = method_data.reindex(MODEL_ORDER)
            means = method_data['acc_mean'].fillna(0)
            stds = method_data['acc_std'].fillna(0)
            ax1.bar(x + j*width - width, means, width,
                    yerr=stds, capsize=3,
                    label=method, color=METHOD_COLORS[method],
                    alpha=0.85, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{dataset} - Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(MODEL_ORDER)
        ax1.legend(loc='lower right', framealpha=0.9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim([0.60, 0.90])
        
        # FID subplot
        ax2 = axes[i, 1]
        for j, method in enumerate(['Exact', 'SISA', 'Weighting']):
            method_data = ds_data[ds_data['method'] == method].set_index('model')
            method_data = method_data.reindex(MODEL_ORDER)
            means = method_data['fid_mean'].fillna(0)
            stds = method_data['fid_std'].fillna(0)
            ax2.bar(x + j*width - width, means, width,
                    yerr=stds, capsize=3,
                    label=method, color=METHOD_COLORS[method],
                    alpha=0.85, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('FID (Feature Importance Distance)')
        ax2.set_title(f'{dataset} - FID')
        ax2.set_xticks(x)
        ax2.set_xticklabels(MODEL_ORDER)
        ax2.legend(loc='upper left', framealpha=0.9)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "figure1_bar_plots.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure1_bar_plots.png", bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {out_path}")


# ============================================================
# Figure 2: Heatmap Cohen's d
# ============================================================
def figure2_heatmaps(summary: pd.DataFrame, forget_size: int = 5):
    """Effect size analysis using Cohen's d for explanation drift."""
    subset = summary[summary['forget_size'] == forget_size]
    
    # Compute Cohen's d matrices
    d_sisa = np.zeros((len(DATASET_ORDER), len(MODEL_ORDER)))
    d_weight = np.zeros((len(DATASET_ORDER), len(MODEL_ORDER)))
    
    for i, ds in enumerate(DATASET_ORDER):
        for j, m in enumerate(MODEL_ORDER):
            row = subset[(subset['dataset'] == ds) & (subset['model'] == m)]
            exact = row[row['method'] == 'Exact']
            sisa = row[row['method'] == 'SISA']
            weight = row[row['method'] == 'Weighting']
            
            if not exact.empty and not sisa.empty:
                e = exact.iloc[0]
                s = sisa.iloc[0]
                d_sisa[i, j] = cohens_d(s['fid_mean'], s['fid_std'],
                                        e['fid_mean'], e['fid_std'])
            if not exact.empty and not weight.empty:
                e = exact.iloc[0]
                w = weight.iloc[0]
                d_weight[i, j] = cohens_d(w['fid_mean'], w['fid_std'],
                                          e['fid_mean'], e['fid_std'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SISA heatmap
    sns.heatmap(d_sisa, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=axes[0],
                xticklabels=MODEL_ORDER, yticklabels=DATASET_ORDER,
                cbar_kws={'label': "Cohen's d (SISA vs Exact)"},
                linewidths=1, linecolor='black',
                annot_kws={'size': 11, 'weight': 'bold'})
    axes[0].set_title("Cohen's d for FID: SISA vs Exact", fontweight='bold')
    axes[0].set_xlabel('Model Architecture', fontweight='bold')
    axes[0].set_ylabel('Dataset', fontweight='bold')
    
    # Weighting heatmap
    sns.heatmap(d_weight, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=axes[1],
                xticklabels=MODEL_ORDER, yticklabels=DATASET_ORDER,
                cbar_kws={'label': "Cohen's d (Weighting vs Exact)"},
                linewidths=1, linecolor='black',
                annot_kws={'size': 11, 'weight': 'bold'})
    axes[1].set_title("Cohen's d for FID: Weighting vs Exact", fontweight='bold')
    axes[1].set_xlabel('Model Architecture', fontweight='bold')
    axes[1].set_ylabel('Dataset', fontweight='bold')
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "figure2_heatmaps.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure2_heatmaps.png", bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {out_path}")


# ============================================================
# Figure 3: Scatter Plot (Accuracy vs FID)
# ============================================================
def figure3_scatter(df: pd.DataFrame, forget_size: int = 5):
    """Scatter plots reveal the decoupling of accuracy and explanation stability."""
    subset = df[df['forget_size'] == forget_size]
    
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(18, 5))
    
    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        ds_data = subset[subset['dataset'] == dataset]
        
        for method in ['Exact', 'SISA', 'Weighting']:
            method_data = ds_data[ds_data['method'] == method]
            ax.scatter(method_data['fid'], method_data['accuracy'],
                       c=METHOD_COLORS[method], label=method,
                       alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('FID (Feature Importance Distance)', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(dataset, fontweight='bold')
        ax.legend(framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Annotation
        ax.text(0.02, 0.95,
                'Accuracy remains stable\nwhile FID varies significantly',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "figure3_scatter.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure3_scatter.png", bbox_inches='tight')
    plt.close()
    print(f"✓ Generated {out_path}")


# ============================================================
# Main Execution
# ============================================================
def main():
    print("=" * 60)
    print("Professional Visualization Generation")
    print("=" * 60)
    
    df = load_all_results()
    summary = compute_summary(df)
    
    print(f"\n✓ Loaded {len(df)} observations")
    print(f"  Generating figures for |Df|=5...")
    
    figure1_bar_plots(summary, forget_size=5)
    figure2_heatmaps(summary, forget_size=5)
    figure3_scatter(df, forget_size=5)
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()