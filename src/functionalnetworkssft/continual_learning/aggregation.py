"""
Multi-seed result aggregation and publication table generation.

Reads experiment CSVs produced by the orchestrator, computes mean +/- std
across seeds, and generates DOC-paper-format tables for publication.
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
}


def load_experiment_csv(csv_path: Path) -> pd.DataFrame:
    """Load the main experiment results CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Parse numeric columns
    for col in ["average_accuracy", "backward_transfer", "forward_transfer"]:
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


def generate_accuracy_table_with_stats(
    agg: pd.DataFrame, model: str, order_group: str = "standard"
) -> pd.DataFrame:
    """Generate DOC paper Table 1/2 format with mean +/- std.

    Args:
        agg: Aggregated DataFrame from aggregate_across_seeds()
        model: Model key to filter on
        order_group: "standard" for orders 1-3, "long_chain" for orders 4-6

    Returns:
        DataFrame with columns: Method, O1, O2, O3, Avg (or O4, O5, O6, Avg)
    """
    orders = STANDARD_ORDERS if order_group == "standard" else LONG_CHAIN_ORDERS
    order_labels = (
        ["O1", "O2", "O3"]
        if order_group == "standard"
        else ["O4", "O5", "O6"]
    )

    methods = list(METHOD_DISPLAY_NAMES.keys())
    rows = []

    for method in methods:
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

    return pd.DataFrame(rows)


def generate_bwt_fwt_table_with_stats(
    agg: pd.DataFrame, model: str
) -> pd.DataFrame:
    """Generate DOC paper Table 3 format with mean +/- std."""
    methods = list(METHOD_DISPLAY_NAMES.keys())
    rows = []

    for method in methods:
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

    return pd.DataFrame(rows)


def generate_per_task_forgetting_table(
    df: pd.DataFrame, model: str, method: str, order: str
) -> Optional[pd.DataFrame]:
    """Generate per-task accuracy breakdown showing forgetting over time.

    Shows accuracy on each task after training on each subsequent task.
    Useful for visualizing the forgetting pattern.
    """
    mask = (
        (df["model"] == model)
        & (df["method"] == method)
        & (df["task_order"] == order)
    )
    rows = df[mask]
    if rows.empty:
        return None

    # Average accuracy matrices across seeds
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

    with open(output_dir / "tables.tex", "w") as f:
        f.write(latex_content)

    # Generate per-task forgetting tables for each method on order_1
    for method in METHOD_DISPLAY_NAMES:
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
    )


if __name__ == "__main__":
    main()
