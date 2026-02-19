"""
Multi-seed result aggregation and publication table generation.

Reads experiment CSVs produced by the orchestrator, computes mean +/- std
across seeds, and generates DOC-paper-format tables for publication.
Includes DOC paper reference numbers and significance testing.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STANDARD_ORDERS = ["order_1", "order_2", "order_3"]
LONG_CHAIN_ORDERS = ["order_4", "order_5", "order_6"]

METHOD_DISPLAY_NAMES = {
    "lora": "LoRA (baseline)",
    "ewc": "EWC",
    "lwf": "LwF",
    "o_lora": "O-LoRA",
    "doc": "DOC",
    "ica_networks": "ICA Networks",
    "ica_lesion": "ICA Lesion",
    "ica_preserve": "ICA Preserve",
    "ica_lesion_antidrift": "ICA Lesion + AD",
    "ica_preserve_antidrift": "ICA Preserve + AD",
}


def load_experiment_csv(csv_path: Path) -> pd.DataFrame:
    """Load the main experiment results CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Parse numeric columns
    for col in ["average_accuracy", "backward_transfer", "forward_transfer"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_baseline_csv(csv_path: Path) -> pd.DataFrame:
    """Load the FWT baselines CSV."""
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    return df


def _format_mean_std(mean: float, std: float, fmt: str = ".1f") -> str:
    """Format mean +/- std for tables."""
    if std == 0 or np.isnan(std):
        return f"{mean:{fmt}}"
    return f"{mean:{fmt}} +/- {std:{fmt}}"


def aggregate_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (model, method, task_order), compute mean and std for metrics."""
    grouped = df.groupby(["model", "method", "task_order"]).agg(
        aa_mean=("average_accuracy", "mean"),
        aa_std=("average_accuracy", "std"),
        bwt_mean=("backward_transfer", "mean"),
        bwt_std=("backward_transfer", "std"),
        fwt_mean=("forward_transfer", "mean"),
        fwt_std=("forward_transfer", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    # Fill NaN std with 0 (single seed)
    grouped["aa_std"] = grouped["aa_std"].fillna(0)
    grouped["bwt_std"] = grouped["bwt_std"].fillna(0)
    grouped["fwt_std"] = grouped["fwt_std"].fillna(0)
    return grouped


def _paired_bootstrap_test(
    values_a: List[float], values_b: List[float], n_bootstrap: int = 10000
) -> float:
    """Paired bootstrap significance test.

    Tests whether values_a is significantly different from values_b.

    Args:
        values_a: Scores for method A across seeds.
        values_b: Scores for method B across seeds.
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        p-value (two-tailed).
    """
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 1.0

    diffs = np.array(values_a) - np.array(values_b)
    observed_diff = np.mean(diffs)
    n = len(diffs)

    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_bootstrap):
        signs = rng.choice([-1, 1], size=n)
        boot_diff = np.mean(diffs * signs)
        if abs(boot_diff) >= abs(observed_diff):
            count += 1

    return count / n_bootstrap


def generate_accuracy_table_with_stats(
    agg: pd.DataFrame,
    model: str,
    order_group: str = "standard",
    include_doc_reference: bool = True,
) -> pd.DataFrame:
    """Generate DOC paper Table 1/2 format with mean +/- std.

    Args:
        agg: Aggregated DataFrame from aggregate_across_seeds()
        model: Model key to filter on
        order_group: "standard" for orders 1-3, "long_chain" for orders 4-6
        include_doc_reference: Whether to include DOC paper reference rows

    Returns:
        DataFrame with columns: Method, O1, O2, O3, Avg (or O4, O5, O6, Avg)
    """
    orders = STANDARD_ORDERS if order_group == "standard" else LONG_CHAIN_ORDERS
    order_labels = (
        ["O1", "O2", "O3"]
        if order_group == "standard"
        else ["O4", "O5", "O6"]
    )

    # Get all methods present in the data
    data_methods = agg[agg["model"] == model]["method"].unique().tolist()
    # Order by METHOD_DISPLAY_NAMES, then alphabetically for unknown
    ordered_methods = [m for m in METHOD_DISPLAY_NAMES if m in data_methods]
    remaining = [m for m in data_methods if m not in ordered_methods]
    ordered_methods.extend(sorted(remaining))

    rows = []

    for method in ordered_methods:
        row = {"Method": METHOD_DISPLAY_NAMES.get(method, method)}
        means = []

        for order, label in zip(orders, order_labels):
            mask = (
                (agg["model"] == model)
                & (agg["method"] == method)
                & (agg["task_order"] == order)
            )
            match = agg[mask]
            if len(match) > 0:
                m = match.iloc[0]["aa_mean"]
                s = match.iloc[0]["aa_std"]
                row[label] = _format_mean_std(m, s)
                means.append(m)
            else:
                row[label] = "-"

        if means:
            avg = np.mean(means)
            row["Avg"] = f"{avg:.1f}"
        else:
            row["Avg"] = "-"

        rows.append(row)

    # Add DOC paper reference rows
    if include_doc_reference:
        try:
            from .reference_results import get_doc_reference_aa

            doc_methods = ["lora", "ewc", "lwf", "o_lora", "doc"]
            for method in doc_methods:
                ref_row = {
                    "Method": f"{METHOD_DISPLAY_NAMES.get(method, method)} (DOC ref)"
                }
                ref_means = []
                for order, label in zip(orders, order_labels):
                    aa = get_doc_reference_aa(model, method, order)
                    if aa is not None:
                        ref_row[label] = f"{aa:.1f}"
                        ref_means.append(aa)
                    else:
                        ref_row[label] = "-"

                if ref_means:
                    ref_row["Avg"] = f"{np.mean(ref_means):.1f}"
                else:
                    ref_row["Avg"] = "-"

                rows.append(ref_row)
        except ImportError:
            pass

    return pd.DataFrame(rows)


def generate_bwt_fwt_table_with_stats(
    agg: pd.DataFrame,
    model: str,
    include_doc_reference: bool = True,
) -> pd.DataFrame:
    """Generate DOC paper Table 3 format with mean +/- std."""
    # Get all methods in data
    data_methods = agg[agg["model"] == model]["method"].unique().tolist()
    ordered_methods = [m for m in METHOD_DISPLAY_NAMES if m in data_methods]
    remaining = [m for m in data_methods if m not in ordered_methods]
    ordered_methods.extend(sorted(remaining))

    rows = []

    for method in ordered_methods:
        row = {"Method": METHOD_DISPLAY_NAMES.get(method, method)}

        for prefix, orders in [("Std", STANDARD_ORDERS), ("Long", LONG_CHAIN_ORDERS)]:
            bwt_vals, fwt_vals = [], []

            for order in orders:
                mask = (
                    (agg["model"] == model)
                    & (agg["method"] == method)
                    & (agg["task_order"] == order)
                )
                match = agg[mask]
                if len(match) > 0:
                    bwt_vals.append(match.iloc[0]["bwt_mean"])
                    fwt_vals.append(match.iloc[0]["fwt_mean"])

            if bwt_vals:
                row[f"{prefix} BWT"] = f"{np.mean(bwt_vals):.2f}"
            else:
                row[f"{prefix} BWT"] = "-"

            if fwt_vals:
                row[f"{prefix} FWT"] = f"{np.mean(fwt_vals):.2f}"
            else:
                row[f"{prefix} FWT"] = "-"

        rows.append(row)

    # Add DOC paper reference rows
    if include_doc_reference:
        try:
            from .reference_results import get_doc_reference_bwt_fwt

            doc_methods = ["lora", "ewc", "lwf", "o_lora", "doc"]
            for method in doc_methods:
                ref = get_doc_reference_bwt_fwt(model, method)
                if ref is None:
                    continue
                ref_row = {
                    "Method": f"{METHOD_DISPLAY_NAMES.get(method, method)} (DOC ref)",
                    "Std BWT": f"{ref['std_bwt']:.2f}",
                    "Std FWT": f"{ref['std_fwt']:.2f}",
                    "Long BWT": f"{ref['long_bwt']:.2f}",
                    "Long FWT": f"{ref['long_fwt']:.2f}",
                }
                rows.append(ref_row)
        except ImportError:
            pass

    return pd.DataFrame(rows)


def generate_lm_eval_table(
    lm_eval_csv: Path,
    model: str,
    include_doc_reference: bool = True,
) -> Optional[pd.DataFrame]:
    """Generate lm-eval comparison table (DOC Table 9 format).

    Args:
        lm_eval_csv: Path to the lm-eval CSV.
        model: Model key to filter on.
        include_doc_reference: Whether to include DOC paper MMLU reference.

    Returns:
        DataFrame with MMLU scores, or None if no data.
    """
    if not lm_eval_csv.exists():
        return None

    df = pd.read_csv(lm_eval_csv)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    model_df = df[df["model"] == model]

    if model_df.empty:
        return None

    rows = []

    # Base model
    base = model_df[model_df["method"] == "base"]
    if not base.empty:
        # Get the primary metric (usually acc or acc_norm)
        for _, row in base.iterrows():
            if "acc" in row["metric"]:
                rows.append({
                    "Condition": "Base model",
                    "MMLU": f"{row['value'] * 100:.1f}" if row["value"] < 1 else f"{row['value']:.1f}",
                })
                break

    # Post-CL results
    methods = model_df[model_df["method"] != "base"]["method"].unique()
    for method in sorted(methods):
        method_df = model_df[model_df["method"] == method]
        acc_rows = method_df[method_df["metric"].str.contains("acc", na=False)]
        if not acc_rows.empty:
            mean_val = acc_rows["value"].mean()
            display = f"{mean_val * 100:.1f}" if mean_val < 1 else f"{mean_val:.1f}"
            rows.append({
                "Condition": f"After CL ({METHOD_DISPLAY_NAMES.get(method, method)})",
                "MMLU": display,
            })

    # DOC paper reference
    if include_doc_reference:
        try:
            from .reference_results import get_doc_reference_mmlu

            ref = get_doc_reference_mmlu(model)
            if ref:
                for condition, score in ref.items():
                    rows.append({
                        "Condition": f"{condition} (DOC ref)",
                        "MMLU": f"{score:.1f}",
                    })
        except ImportError:
            pass

    return pd.DataFrame(rows) if rows else None


def generate_significance_table(
    df: pd.DataFrame,
    model: str,
    baseline_method: str = "lora",
    order_group: str = "standard",
) -> Optional[pd.DataFrame]:
    """Generate paired bootstrap significance tests vs baseline.

    Args:
        df: Raw experiment results DataFrame (not aggregated).
        model: Model key.
        baseline_method: Method to compare against.
        order_group: "standard" or "long_chain".

    Returns:
        DataFrame with p-values, or None if insufficient data.
    """
    orders = STANDARD_ORDERS if order_group == "standard" else LONG_CHAIN_ORDERS
    model_df = df[df["model"] == model]

    methods = [m for m in model_df["method"].unique() if m != baseline_method]
    if not methods:
        return None

    rows = []
    for method in methods:
        baseline_scores = []
        method_scores = []

        for order in orders:
            bl = model_df[
                (model_df["method"] == baseline_method) & (model_df["task_order"] == order)
            ]["average_accuracy"].tolist()
            mt = model_df[
                (model_df["method"] == method) & (model_df["task_order"] == order)
            ]["average_accuracy"].tolist()

            # Match seeds
            min_len = min(len(bl), len(mt))
            baseline_scores.extend(bl[:min_len])
            method_scores.extend(mt[:min_len])

        if len(baseline_scores) >= 2:
            p_val = _paired_bootstrap_test(method_scores, baseline_scores)
            diff = np.mean(method_scores) - np.mean(baseline_scores)
            sig = "*" if p_val < 0.05 else ""
            rows.append({
                "Method": METHOD_DISPLAY_NAMES.get(method, method),
                "AA Diff vs LoRA": f"{diff:+.1f}{sig}",
                "p-value": f"{p_val:.3f}",
            })

    return pd.DataFrame(rows) if rows else None


def generate_per_task_forgetting_table(
    df: pd.DataFrame, model: str, method: str, order: str
) -> Optional[pd.DataFrame]:
    """Generate per-task accuracy breakdown showing forgetting over time."""
    mask = (
        (df["model"] == model)
        & (df["method"] == method)
        & (df["task_order"] == order)
    )
    rows = df[mask]
    if rows.empty:
        return None

    matrices = []
    for _, row in rows.iterrows():
        try:
            matrix = json.loads(row["accuracy_matrix"])
            matrices.append(np.array(matrix))
        except (json.JSONDecodeError, TypeError):
            continue

    if not matrices:
        return None

    avg_matrix = np.mean(matrices, axis=0)
    from .task_data.config import get_task_order

    task_names = get_task_order(order)
    columns = [f"After T{i+1}" for i in range(avg_matrix.shape[1])]
    result = pd.DataFrame(avg_matrix, columns=columns, index=task_names)
    result.index.name = "Task"
    return result


def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to LaTeX table."""
    latex = df.to_latex(index=False, escape=False, column_format="l" + "c" * (len(df.columns) - 1))
    return (
        f"\\begin{{table}}[h]\n\\centering\n\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n{latex}\\end{{table}}\n"
    )


def generate_all_tables(
    results_csv: Path,
    baseline_csv: Path,
    model: str,
    output_dir: Path,
    lm_eval_csv: Optional[Path] = None,
) -> None:
    """Generate all publication tables from results CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_experiment_csv(results_csv)
    agg = aggregate_across_seeds(df)

    # Accuracy tables
    std_table = generate_accuracy_table_with_stats(agg, model, "standard")
    long_table = generate_accuracy_table_with_stats(agg, model, "long_chain")
    bwt_fwt = generate_bwt_fwt_table_with_stats(agg, model)

    # Print summary
    print(f"\nResults for {model}")
    print("=" * 70)
    print("\n## Average Accuracy (AA) - Standard Benchmark")
    print(std_table.to_markdown(index=False))
    print("\n## Average Accuracy (AA) - Long Chain")
    print(long_table.to_markdown(index=False))
    print("\n## Backward/Forward Transfer")
    print(bwt_fwt.to_markdown(index=False))

    # Export CSVs
    std_table.to_csv(output_dir / "accuracy_table_standard.csv", index=False)
    long_table.to_csv(output_dir / "accuracy_table_long_chain.csv", index=False)
    bwt_fwt.to_csv(output_dir / "bwt_fwt_table.csv", index=False)
    agg.to_csv(output_dir / "aggregated_results.csv", index=False)

    # Export LaTeX
    latex_content = ""
    latex_content += _to_latex(
        std_table,
        f"Average Accuracy (\\%) on Standard CL Benchmark ({model})",
        f"tab:aa_std_{model}",
    )
    latex_content += "\n"
    latex_content += _to_latex(
        long_table,
        f"Average Accuracy (\\%) on Long Chain ({model})",
        f"tab:aa_long_{model}",
    )
    latex_content += "\n"
    latex_content += _to_latex(
        bwt_fwt,
        f"Backward and Forward Transfer ({model})",
        f"tab:bwt_fwt_{model}",
    )

    # Significance testing
    sig_std = generate_significance_table(df, model, "lora", "standard")
    if sig_std is not None:
        print("\n## Significance vs LoRA (Standard)")
        print(sig_std.to_markdown(index=False))
        sig_std.to_csv(output_dir / "significance_standard.csv", index=False)
        latex_content += "\n"
        latex_content += _to_latex(
            sig_std,
            f"Significance Tests vs LoRA Baseline ({model})",
            f"tab:sig_{model}",
        )

    sig_long = generate_significance_table(df, model, "lora", "long_chain")
    if sig_long is not None:
        print("\n## Significance vs LoRA (Long Chain)")
        print(sig_long.to_markdown(index=False))
        sig_long.to_csv(output_dir / "significance_long_chain.csv", index=False)

    # lm-eval table
    if lm_eval_csv is not None:
        lm_eval_table = generate_lm_eval_table(lm_eval_csv, model)
        if lm_eval_table is not None:
            print("\n## MMLU Scores")
            print(lm_eval_table.to_markdown(index=False))
            lm_eval_table.to_csv(output_dir / "mmlu_comparison.csv", index=False)
            latex_content += "\n"
            latex_content += _to_latex(
                lm_eval_table,
                f"MMLU Scores Before/After CL ({model})",
                f"tab:mmlu_{model}",
            )

    with open(output_dir / "tables.tex", "w") as f:
        f.write(latex_content)

    # Generate per-task forgetting tables for each method on order_1
    all_methods = df[df["model"] == model]["method"].unique()
    for method in all_methods:
        ft = generate_per_task_forgetting_table(df, model, method, "order_1")
        if ft is not None:
            ft.to_csv(
                output_dir / f"forgetting_{method}_order_1.csv"
            )

    # Summary markdown
    with open(output_dir / "summary.md", "w") as f:
        f.write(f"# Continual Learning Results: {model}\n\n")
        f.write(f"Seeds per configuration: {agg['n_seeds'].max()}\n\n")
        f.write("## Average Accuracy (AA) - Standard Benchmark\n\n")
        f.write(std_table.to_markdown(index=False))
        f.write("\n\n## Average Accuracy (AA) - Long Chain\n\n")
        f.write(long_table.to_markdown(index=False))
        f.write("\n\n## Backward/Forward Transfer\n\n")
        f.write(bwt_fwt.to_markdown(index=False))
        if sig_std is not None:
            f.write("\n\n## Significance vs LoRA (Standard)\n\n")
            f.write(sig_std.to_markdown(index=False))
        f.write("\n")

    logger.info(f"Tables saved to {output_dir}")


def main():
    """CLI entry point for result aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate CL experiment results and generate publication tables"
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="Path to experiment results CSV",
    )
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default=None,
        help="Path to FWT baselines CSV",
    )
    parser.add_argument(
        "--lm_eval_csv",
        type=str,
        default=None,
        help="Path to lm-eval results CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key to filter results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments/continual_learning/results/tables",
        help="Output directory for tables",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    generate_all_tables(
        results_csv=Path(args.results_csv),
        baseline_csv=Path(args.baseline_csv) if args.baseline_csv else Path("/dev/null"),
        model=args.model,
        output_dir=Path(args.output_dir),
        lm_eval_csv=Path(args.lm_eval_csv) if args.lm_eval_csv else None,
    )


if __name__ == "__main__":
    main()
