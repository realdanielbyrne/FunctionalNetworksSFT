#!/usr/bin/env python3
"""
Generate comparison tables in DOC paper format.

Usage:
    python generate_tables.py --results_dir ./results --model llama-7b
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(results_dir: Path) -> dict:
    """Load all result files from directory."""
    results = {}
    for json_file in results_dir.glob("**/*.json"):
        with open(json_file) as f:
            data = json.load(f)
        key = f"{data['model']}_{data['method']}_{data['task_order']}"
        results[key] = data
    return results


def generate_accuracy_table(results: dict, model: str = "llama-7b") -> pd.DataFrame:
    """Generate accuracy table in DOC paper format."""
    methods = ["lora", "ewc", "lwf", "o_lora", "doc", "ica_networks"]
    standard_orders = ["order_1", "order_2", "order_3"]
    long_orders = ["order_4", "order_5", "order_6"]

    rows = []
    for method in methods:
        row = {"Method": method.upper()}

        std_accs = []
        for i, order in enumerate(standard_orders):
            key = f"{model}_{method}_{order}"
            if key in results:
                acc = results[key]["average_accuracy"]
                row[f"O{i + 1}"] = acc
                std_accs.append(acc)

        if std_accs:
            row["Std Avg"] = sum(std_accs) / len(std_accs)

        long_accs = []
        for i, order in enumerate(long_orders):
            key = f"{model}_{method}_{order}"
            if key in results:
                acc = results[key]["average_accuracy"]
                row[f"O{i + 4}"] = acc
                long_accs.append(acc)

        if long_accs:
            row["Long Avg"] = sum(long_accs) / len(long_accs)

        rows.append(row)

    return pd.DataFrame(rows)


def generate_bwt_fwt_table(results: dict, model: str = "llama-7b") -> pd.DataFrame:
    """Generate BWT/FWT table in DOC paper format."""
    methods = ["lora", "ewc", "lwf", "o_lora", "doc", "ica_networks"]
    standard_orders = ["order_1", "order_2", "order_3"]
    long_orders = ["order_4", "order_5", "order_6"]

    rows = []
    for method in methods:
        row = {"Method": method.upper()}

        std_bwt, std_fwt = [], []
        for order in standard_orders:
            key = f"{model}_{method}_{order}"
            if key in results:
                std_bwt.append(results[key]["backward_transfer"])
                std_fwt.append(results[key]["forward_transfer"])

        if std_bwt:
            row["Std BWT"] = sum(std_bwt) / len(std_bwt)
            row["Std FWT"] = sum(std_fwt) / len(std_fwt)

        long_bwt, long_fwt = [], []
        for order in long_orders:
            key = f"{model}_{method}_{order}"
            if key in results:
                long_bwt.append(results[key]["backward_transfer"])
                long_fwt.append(results[key]["forward_transfer"])

        if long_bwt:
            row["Long BWT"] = sum(long_bwt) / len(long_bwt)
            row["Long FWT"] = sum(long_fwt) / len(long_fwt)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate CL comparison tables")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="llama-7b")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = load_results(Path(args.results_dir))

    acc_table = generate_accuracy_table(results, args.model)
    bwt_fwt_table = generate_bwt_fwt_table(results, args.model)

    print(f"\n{'=' * 60}")
    print(f"Results for {args.model}")
    print(f"{'=' * 60}")

    print("\n## Average Accuracy (AA)")
    print(acc_table.to_markdown(index=False, floatfmt=".1f"))

    print("\n## Backward/Forward Transfer (BWT/FWT)")
    print(bwt_fwt_table.to_markdown(index=False, floatfmt=".2f"))

    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        acc_table.to_csv(output_path / "accuracy_table.csv", index=False)
        bwt_fwt_table.to_csv(output_path / "bwt_fwt_table.csv", index=False)
        print(f"\nTables saved to {output_path}")


if __name__ == "__main__":
    main()

