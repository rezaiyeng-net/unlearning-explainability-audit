"""
Professional Statistical Analysis and LaTeX Table Generation
Analyzes unlearning experiment results and generates publication-ready tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = BASE_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True)

# Standardize names
DATASET_MAP = {
    'german': 'German Credit', 'German Credit': 'German Credit',
    'adult': 'Adult', 'Adult': 'Adult',
    'heart': 'Heart Disease', 'Heart Disease': 'Heart Disease'
}
MODEL_MAP = {
    'LR': 'LR', 'RF': 'RF', 'GB': 'GB', 'MLP': 'MLP'
}
MODEL_ORDER = ['LR', 'RF', 'GB', 'MLP']
DATASET_ORDER = ['German Credit', 'Adult', 'Heart Disease']


# ============================================================
# Data Loading
# ============================================================
def load_all_results() -> pd.DataFrame:
    """Load and standardize results from both experiment files."""
    dfs = []
    for fname in ['part1_results.csv', 'part2_results.csv']:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            df['dataset'] = df['dataset'].map(DATASET_MAP)
            df['model'] = df['model'].map(MODEL_MAP)
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    # Ensure MIA column exists
    if 'mia' not in combined.columns:
        combined['mia'] = np.nan
    return combined


# ============================================================
# Statistical Functions
# ============================================================
def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for each group."""
    stats = df.groupby(['dataset', 'model', 'forget_size', 'method']).agg(
        acc_mean=('accuracy', 'mean'),
        acc_std=('accuracy', 'std'),
        fid_mean=('fid', 'mean'),
        fid_std=('fid', 'std'),
        mia_mean=('mia', 'mean'),
        mia_std=('mia', 'std'),
        n=('accuracy', 'count')
    ).reset_index()
    return stats


def cohens_d(mean1: float, std1: float, n1: int,
             mean2: float, std2: float, n2: int) -> Tuple[float, str]:
    """Compute Cohen's d and interpret effect size."""
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    if pooled_std == 0:
        return 0.0, "Neg."
    d = (mean1 - mean2) / pooled_std
    abs_d = abs(d)
    if abs_d < 0.2:
        label = "Neg."
    elif abs_d < 0.5:
        label = "Small"
    elif abs_d < 0.8:
        label = "Med."
    else:
        label = "Large"
    return d, label


def add_cohens_d(stats: pd.DataFrame) -> pd.DataFrame:
    """Add Cohen's d columns relative to Exact method."""
    rows = []
    for (ds, m, fs), group in stats.groupby(['dataset', 'model', 'forget_size']):
        exact = group[group['method'] == 'Exact']
        if exact.empty:
            continue
        e = exact.iloc[0]
        for _, row in group.iterrows():
            new_row = row.copy()
            if row['method'] == 'Exact':
                new_row['d_acc'] = np.nan
                new_row['d_fid'] = np.nan
                new_row['d_acc_label'] = '---'
                new_row['d_fid_label'] = '---'
            else:
                d_acc, lab_acc = cohens_d(
                    row['acc_mean'], row['acc_std'], row['n'],
                    e['acc_mean'], e['acc_std'], e['n']
                )
                d_fid, lab_fid = cohens_d(
                    row['fid_mean'], row['fid_std'], row['n'],
                    e['fid_mean'], e['fid_std'], e['n']
                )
                new_row['d_acc'] = d_acc
                new_row['d_fid'] = d_fid
                new_row['d_acc_label'] = f"{d_acc:.2f} ({lab_acc})"
                new_row['d_fid_label'] = f"{d_fid:.2f} ({lab_fid})"
            rows.append(new_row)
    return pd.DataFrame(rows)


# ============================================================
# LaTeX Table Generators
# ============================================================
def format_val(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def generate_hero_table(stats: pd.DataFrame,
                        dataset: str = 'German Credit',
                        forget_size: int = 5) -> str:
    """Generate the Hero Table for main manuscript."""
    subset = stats[(stats['dataset'] == dataset) &
                   (stats['forget_size'] == forget_size)]
    
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Statistical comparison of unlearning methods on "
        f"{dataset} across four model architectures "
        r"(Mean $\pm$ Std over 15 random seeds, $|D_f|=$" + f"{forget_size}" + r").}",
        r"\label{tab:hero_table}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Method} & \textbf{Accuracy} & "
        r"\textbf{FID} & \textbf{Cohen's $d$ (Acc)} & "
        r"\textbf{Cohen's $d$ (FID)} \\",
        r"\midrule"
    ]
    
    for model in MODEL_ORDER:
        model_rows = subset[subset['model'] == model]
        if model_rows.empty:
            continue
        for i, (_, row) in enumerate(model_rows.iterrows()):
            model_name = model if i == 0 else ''
            acc_str = format_val(row['acc_mean'], row['acc_std'], 3)
            fid_str = format_val(row['fid_mean'], row['fid_std'], 4)
            d_acc = row['d_acc_label'] if pd.notna(row['d_acc']) else '---'
            d_fid = row['d_fid_label'] if pd.notna(row['d_fid']) else '---'
            lines.append(
                f"{model_name} & {row['method']} & {acc_str} & "
                f"{fid_str} & {d_acc} & {d_fid} \\"
            )
        lines.append(r"\midrule")
    
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_comprehensive_table(stats: pd.DataFrame,
                                  forget_size: int = 5) -> str:
    """Generate comprehensive summary table."""
    subset = stats[stats['forget_size'] == forget_size]
    
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Comprehensive statistical comparison across datasets "
        r"and models (Mean $\pm$ Std over 15 random seeds, $|D_f|=$" + f"{forget_size}" + r").}",
        r"\label{tab:comprehensive}",
        r"\small",
        r"\begin{tabular}{llllccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Model} & \textbf{Method} & "
        r"\textbf{Accuracy} & \textbf{FID} & \textbf{MIA} & "
        r"\textbf{Cohen's $d$ (FID)} \\",
        r"\midrule"
    ]
    
    for dataset in DATASET_ORDER:
        ds_subset = subset[subset['dataset'] == dataset]
        if ds_subset.empty:
            continue
        for model in MODEL_ORDER:
            model_rows = ds_subset[ds_subset['model'] == model]
            if model_rows.empty:
                continue
            for i, (_, row) in enumerate(model_rows.iterrows()):
                ds_name = dataset if (i == 0 and 
                    model == model_rows.iloc[0]['model']) else ''
                model_name = model if i == 0 else ''
                acc_str = format_val(row['acc_mean'], row['acc_std'], 4)
                fid_str = format_val(row['fid_mean'], row['fid_std'], 6)
                mia_str = format_val(row['mia_mean'], row['mia_std'], 4)
                d_fid = row['d_fid_label'] if pd.notna(row['d_fid']) else '---'
                lines.append(
                    f"{ds_name} & {model_name} & {row['method']} & "
                    f"{acc_str} & {fid_str} & {mia_str} & {d_fid} \\"
                )
            lines.append(r"\cmidrule(lr){2-7}")
    
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_dataset_table(stats: pd.DataFrame,
                           dataset: str,
                           forget_size: int = 5) -> str:
    """Generate per-dataset table."""
    subset = stats[(stats['dataset'] == dataset) &
                   (stats['forget_size'] == forget_size)]
    
    safe_name = dataset.lower().replace(' ', '_')
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Statistical comparison on " + dataset +
        r" ($|D_f|=$" + f"{forget_size}" + r").}",
        r"\label{tab:" + safe_name + f"_{forget_size}" + r"}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Method} & \textbf{Accuracy} & "
        r"\textbf{FID} & \textbf{Cohen's $d$ (Acc)} & "
        r"\textbf{Cohen's $d$ (FID)} \\",
        r"\midrule"
    ]
    
    for model in MODEL_ORDER:
        model_rows = subset[subset['model'] == model]
        if model_rows.empty:
            continue
        for i, (_, row) in enumerate(model_rows.iterrows()):
            model_name = model if i == 0 else ''
            acc_str = format_val(row['acc_mean'], row['acc_std'], 3)
            fid_str = format_val(row['fid_mean'], row['fid_std'], 4)
            d_acc = row['d_acc_label'] if pd.notna(row['d_acc']) else '---'
            d_fid = row['d_fid_label'] if pd.notna(row['d_fid']) else '---'
            lines.append(
                f"{model_name} & {row['method']} & {acc_str} & "
                f"{fid_str} & {d_acc} & {d_fid} \\"
            )
        lines.append(r"\midrule")
    
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_appendix_index() -> str:
    """Generate appendix LaTeX file with all tables."""
    lines = [
        r"\appendix",
        r"\section{Additional Experimental Results}",
        r"\label{app:additional_results}",
        "",
        r"To demonstrate the generalizability of our findings beyond "
        r"the German Credit dataset, we present comprehensive statistical "
        r"comparisons for the Adult Census and Heart Disease datasets.",
        "",
        r"\subsection{Adult Census Dataset}",
        r"\input{tables/tab_adult_5}",
        "",
        r"\subsection{Heart Disease Dataset}",
        r"\input{tables/tab_heart_disease_5}",
        "",
        r"\subsection{German Credit with $|D_f|=10$}",
        r"\input{tables/tab_german_credit_10}",
        "",
        r"\subsection{Comprehensive Summary}",
        r"\input{tables/tab_comprehensive_5}",
    ]
    return "\n".join(lines)


# ============================================================
# Main Execution
# ============================================================
def main():
    print("=" * 60)
    print("Statistical Analysis and Table Generation")
    print("=" * 60)
    
    # Load data
    df = load_all_results()
    print(f"\n✓ Loaded {len(df)} total observations")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Forget sizes: {sorted(df['forget_size'].unique())}")
    
    # Compute statistics
    stats = compute_descriptive_stats(df)
    stats = add_cohens_d(stats)
    
    # Save summary CSV
    summary_path = RESULTS_DIR / "summary_statistics.csv"
    stats.to_csv(summary_path, index=False)
    print(f"\n✓ Summary statistics saved to {summary_path}")
    
    # Generate tables
    tables = {
        'tab_hero_5.tex': generate_hero_table(stats, 'German Credit', 5),
        'tab_comprehensive_5.tex': generate_comprehensive_table(stats, 5),
        'tab_german_credit_5.tex': generate_dataset_table(stats, 'German Credit', 5),
        'tab_german_credit_10.tex': generate_dataset_table(stats, 'German Credit', 10),
        'tab_adult_5.tex': generate_dataset_table(stats, 'Adult', 5),
        'tab_adult_10.tex': generate_dataset_table(stats, 'Adult', 10),
        'tab_heart_disease_5.tex': generate_dataset_table(stats, 'Heart Disease', 5),
        'tab_heart_disease_10.tex': generate_dataset_table(stats, 'Heart Disease', 10),
    }
    
    for fname, content in tables.items():
        fpath = TABLES_DIR / fname
        fpath.write_text(content, encoding='utf-8')
        print(f"✓ Generated {fpath}")
    
    # Generate appendix index
    appendix_path = TABLES_DIR / "appendix_tables.tex"
    appendix_path.write_text(generate_appendix_index(), encoding='utf-8')
    print(f"✓ Generated {appendix_path}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()