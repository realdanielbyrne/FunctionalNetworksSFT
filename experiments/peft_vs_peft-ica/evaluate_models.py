#!/usr/bin/env python3
"""
Evaluate and compare Experiments A (PEFT-only), B (PEFT+ICA lesion), and C (PEFT+ICA preserve).

Features:
- Loads trained models from experiment output directories (prefers merged_model if present)
- Uses a common evaluation split from the configured dataset to ensure fairness
- Computes metrics:
  * Per-example Negative Log-Likelihood (NLL) and aggregated Perplexity (PPL)
  * Sentence-level BLEU (sacrebleu) if available
  * Length ratio and average response length
  * Optional ROUGE-L via evaluate if available (aggregate only)
- Statistical testing: bootstrap 95% CI and p-values for pairwise NLL comparisons
- Visualizations: bar charts with error bars saved as PNG
- Outputs JSON files per experiment and a combined comparison plus a Markdown summary
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional metrics
try:
    import evaluate as hf_evaluate  # For ROUGE
except Exception:
    hf_evaluate = None

try:
    import sacrebleu  # For BLEU
except Exception:
    sacrebleu = None


LOGGER = logging.getLogger(__name__)


def find_best_model_dir(exp_output_dir: str) -> Optional[str]:
    """Pick a model directory to evaluate. Prefer merged_model/, fall back to final_model/.
    For multi-run experiments (B/C), if run_* subdirs exist, return the list of per-run model dirs.
    """
    if not os.path.exists(exp_output_dir):
        return None

    # Detect multi-run structure
    run_dirs = [
        os.path.join(exp_output_dir, d)
        for d in os.listdir(exp_output_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(exp_output_dir, d))
    ]
    if run_dirs:
        # Return comma-joined list of model dirs to signal multi-run
        selected = []
        for r in sorted(run_dirs):
            cand = None
            for sub in ["merged_model", "final_model"]:
                p = os.path.join(r, sub)
                if os.path.exists(p):
                    cand = p
                    break
            if cand is None:
                # Also allow adapter as fallback
                p = os.path.join(r, "adapter")
                if os.path.exists(p):
                    cand = p
            if cand:
                selected.append(cand)
        return ",".join(selected) if selected else None

    # Single-run structure
    for sub in ["merged_model", "final_model", "adapter"]:
        p = os.path.join(exp_output_dir, sub)
        if os.path.exists(p):
            return p

    # If none found, maybe the output dir itself is a model dir
    return exp_output_dir if any(
        os.path.exists(os.path.join(exp_output_dir, f))
        for f in ["config.json", "adapter_config.json", "pytorch_model.bin", "model.safetensors", "adapter_model.safetensors"]
    ) else None


def load_tokenizer_and_model(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    # Ensure pad token for safe batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tok, model, device


def format_prompt(example: Dict[str, str]) -> Tuple[str, str]:
    """Build a simple instruction→response pair from Dolly 15k-style example.
    Returns (prompt, reference_response).
    """
    # Try common Dolly fields; fall back to generic fields if needed
    prompt_parts = []
    for key in ["instruction", "context", "input", "question"]:
        if key in example and example[key]:
            prompt_parts.append(str(example[key]).strip())
    prompt = "\n\n".join(prompt_parts) if prompt_parts else str(example)

    # Reference answer
    for key in ["response", "output", "answer"]:
        if key in example and example[key]:
            return prompt, str(example[key]).strip()
    return prompt, ""


def build_eval_split(dataset_name_or_path: str, test_size: float, max_samples: Optional[int]) -> List[Dict[str, str]]:
    ds = load_dataset(dataset_name_or_path)
    # Use train split or merge if multiple
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    all_data = ds[split_name]

    # Deterministic subset
    idx = np.random.RandomState(42).permutation(len(all_data))
    n_test = int(len(all_data) * test_size)
    idx = idx[: n_test if n_test > 0 else min(1024, len(all_data))]

    examples = []
    for i in idx:
        ex = all_data[int(i)]
        prompt, ref = format_prompt(ex)
        examples.append({"prompt": prompt, "reference": ref})
        if max_samples and len(examples) >= max_samples:
            break
    return examples


def compute_per_example_nll(tok, model, device, prompts: List[str], references: List[str], max_new_tokens: int = 128) -> Tuple[np.ndarray, List[str]]:
    """Compute per-example NLL over the generated continuation tokens when teacher-forcing on reference.
    Also returns generated outputs for optional generation metrics.
    """
    nlls = []
    generations = []

    for prompt, reference in tqdm(zip(prompts, references), total=len(prompts), desc="Evaluating"):
        # Build input ids as prompt + reference, compute loss over reference tokens only
        with torch.no_grad():
            enc_prompt = tok(prompt, return_tensors="pt")
            enc_full = tok(prompt + tok.eos_token + reference, return_tensors="pt")

            input_ids = enc_full["input_ids"].to(device)
            attn = enc_full["attention_mask"].to(device)

            prompt_len = enc_prompt["input_ids"].shape[1] + 1  # include eos
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100  # ignore prompt tokens

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            # Average loss over labeled tokens
            loss = out.loss.detach().item()
            # Convert to per-token NLL: loss is already mean over labeled tokens
            nlls.append(loss)

            # Greedy generate for qualitative metrics
            gen_ids = model.generate(
                **tok(prompt, return_tensors="pt").to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)
            # Try to strip the prompt from generated text
            if gen_text.startswith(prompt):
                gen_text = gen_text[len(prompt) :].strip()
            generations.append(gen_text)

    return np.array(nlls, dtype=np.float32), generations


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    diffs = []
    n = min(len(a), len(b))
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs.append(float(np.mean(a[idx]) - np.mean(b[idx])))
    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = float(min(np.mean(diffs <= 0), np.mean(diffs >= 0)) * 2)
    return {"diff_mean": float(np.mean(diffs)), "ci_low": float(ci_low), "ci_high": float(ci_high), "p_value": p_value}


def sentence_bleu_scores(references: List[str], predictions: List[str]) -> np.ndarray:
    if sacrebleu is None:
        return np.array([])
    scores = []
    for ref, pred in zip(references, predictions):
        try:
            s = sacrebleu.sentence_bleu(pred, [ref]).score
        except Exception:
            s = 0.0
        scores.append(s)
    return np.array(scores, dtype=np.float32)


def rougeL_aggregate(references: List[str], predictions: List[str]) -> Optional[Dict[str, float]]:
    if hf_evaluate is None:
        return None
    try:
        rouge = hf_evaluate.load("rouge")
        res = rouge.compute(predictions=predictions, references=references)
        return {"rougeL": float(res.get("rougeL", 0.0))}
    except Exception:
        return None


def plot_bars(output_dir: str, title: str, labels: List[str], values: List[float], y_label: str, filename: str, errors: Optional[List[float]] = None, invert: bool = False):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        LOGGER.warning("matplotlib not available; skipping plot %s", filename)
        return
    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    y = values
    if invert:
        y = [-v for v in values]
    ax.bar(x, y, yerr=errors if errors is not None else None, capsize=5)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def evaluate_model_dir(model_dir: str, eval_examples: List[Dict[str, str]], max_new_tokens: int = 128) -> Dict[str, any]:
    tok, model, device = load_tokenizer_and_model(model_dir)
    prompts = [e["prompt"] for e in eval_examples]
    refs = [e["reference"] for e in eval_examples]

    nlls, generations = compute_per_example_nll(tok, model, device, prompts, refs)
    ppl = float(np.exp(np.mean(nlls))) if len(nlls) else float("inf")

    bleu_scores = sentence_bleu_scores(refs, generations)
    rouge_res = rougeL_aggregate(refs, generations)
    len_ratios = np.array([
        (len(g.split()) / max(1, len(r.split()))) if r else float("nan") for g, r in zip(generations, refs)
    ], dtype=np.float32)

    return {
        "nll_per_example": nlls.tolist(),
        "ppl": ppl,
        "bleu_per_example": bleu_scores.tolist() if len(bleu_scores) else None,
        "bleu_mean": float(np.mean(bleu_scores)) if len(bleu_scores) else None,
        "rouge": rouge_res,
        "length_ratio_mean": float(np.nanmean(len_ratios)),
        "avg_gen_length": float(np.mean([len(g.split()) for g in generations])) if generations else 0.0,
    }


def aggregate_multi_run(run_results: List[Dict[str, any]]) -> Dict[str, any]:
    metrics = [r["ppl"] for r in run_results]
    return {
        "runs": run_results,
        "ppl_mean": float(np.mean(metrics)),
        "ppl_std": float(np.std(metrics)),
        "best_ppl": float(np.min(metrics)),
        "best_run_index": int(np.argmin(metrics)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Experiments A, B, C")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="experiments/peft_vs_peft-ica/evaluation_results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Locate model directories
    exp_dirs = {
        "A": "experiments/peft_vs_peft-ica/experiment_a_peft_only/output",
        "B": "experiments/peft_vs_peft-ica/experiment_b_peft_ica/output",
        "C": "experiments/peft_vs_peft-ica/experiment_c_peft_ica_preserve/output",
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Build a common evaluation split
    dataset_name = "databricks/databricks-dolly-15k"
    eval_examples = build_eval_split(dataset_name, args.test_size, args.max_samples)

    # Evaluate each experiment
    results: Dict[str, any] = {}
    per_exp_model_dirs: Dict[str, List[str]] = {}

    for label, out_dir in exp_dirs.items():
        model_path = find_best_model_dir(out_dir)
        if model_path is None:
            logging.warning("No model found for experiment %s in %s", label, out_dir)
            continue
        if "," in model_path:
            model_dirs = model_path.split(",")
            per_exp_model_dirs[label] = model_dirs
            run_results = []
            for i, md in enumerate(model_dirs, 1):
                logging.info("Evaluating %s run %d: %s", label, i, md)
                run_res = evaluate_model_dir(md, eval_examples)
                run_results.append(run_res)
            results[label] = aggregate_multi_run(run_results)
        else:
            per_exp_model_dirs[label] = [model_path]
            logging.info("Evaluating %s: %s", label, model_path)
            results[label] = evaluate_model_dir(model_path, eval_examples)

        # Save per-experiment JSON
        with open(os.path.join(args.output_dir, f"experiment_{label}_results.json"), "w") as f:
            json.dump(results[label], f, indent=2)

    # Statistical tests: compare A vs best-of-B, A vs best-of-C, and B vs C (best vs best)
    stats = {}
    def best_nlls(res):
        if "runs" in res:
            # pick best run's nlls
            idx = res.get("best_run_index", 0)
            return np.array(res["runs"][idx]["nll_per_example"], dtype=np.float32)
        return np.array(res.get("nll_per_example", []), dtype=np.float32)

    if "A" in results and "B" in results:
        stats["A_vs_B"] = bootstrap_diff_ci(best_nlls(results["A"]), best_nlls(results["B"]))
    if "A" in results and "C" in results:
        stats["A_vs_C"] = bootstrap_diff_ci(best_nlls(results["A"]), best_nlls(results["C"]))
    if "B" in results and "C" in results:
        stats["B_vs_C"] = bootstrap_diff_ci(best_nlls(results["B"]), best_nlls(results["C"]))

    with open(os.path.join(args.output_dir, "model_comparison.json"), "w") as f:
        json.dump({"results": results, "stats": stats}, f, indent=2)

    # Plots
    labels = []
    ppl_vals = []
    ppl_err = []
    for label in ["A", "B", "C"]:
        if label not in results:
            continue
        if "runs" in results[label]:
            labels.append(label)
            ppl_vals.append(results[label]["ppl_mean"])
            ppl_err.append(results[label]["ppl_std"])
        else:
            labels.append(label)
            ppl_vals.append(results[label]["ppl"])
            ppl_err.append(0.0)

    plot_bars(args.output_dir, "Perplexity (lower is better)", labels, ppl_vals, "PPL", "perplexity.png", errors=ppl_err)

    # Markdown summary
    md_lines = [
        "# Evaluation Summary",
        "",
        "## Experiments",
        "- A: PEFT-only",
        "- B: PEFT+ICA (lesion) – 3 runs (components [0], [0,1], [0,1,2])",
        "- C: PEFT+ICA (preserve) – 3 runs (components [0], [0,1], [0,1,2])",
        "",
        "## Key Metric: Perplexity",
    ]

    for label in ["A", "B", "C"]:
        if label not in results:
            continue
        if "runs" in results[label]:
            md_lines.append(f"- {label}: mean PPL={results[label]['ppl_mean']:.3f} ± {results[label]['ppl_std']:.3f} (best={results[label]['best_ppl']:.3f})")
        else:
            md_lines.append(f"- {label}: PPL={results[label]['ppl']:.3f}")

    md_lines.append("")
    md_lines.append("## Pairwise NLL Statistical Tests (bootstrap 95% CI)")
    for k, v in stats.items():
        md_lines.append(f"- {k}: diff_mean={v['diff_mean']:.4f} (A-B < 0 favors first), 95% CI=({v['ci_low']:.4f}, {v['ci_high']:.4f}), p={v['p_value']:.4f}")

    md_lines.append("")
    md_lines.append("## Generation Metrics (if available)")
    for label in ["A", "B", "C"]:
        if label not in results:
            continue
        res = results[label]
        if "runs" in res:
            best_idx = res.get("best_run_index", 0)
            best = res["runs"][best_idx]
            bleu = best.get("bleu_mean")
            rouge = best.get("rouge", {}).get("rougeL") if best.get("rouge") else None
            lr = best.get("length_ratio_mean")
            md_lines.append(f"- {label} (best run): BLEU={bleu if bleu is not None else 'n/a'}, ROUGE-L={rouge if rouge is not None else 'n/a'}, LenRatio={lr:.3f}")
        else:
            bleu = res.get("bleu_mean")
            rouge = res.get("rouge", {}).get("rougeL") if res.get("rouge") else None
            lr = res.get("length_ratio_mean")
            md_lines.append(f"- {label}: BLEU={bleu if bleu is not None else 'n/a'}, ROUGE-L={rouge if rouge is not None else 'n/a'}, LenRatio={lr:.3f}")

    with open(os.path.join(args.output_dir, "evaluation_summary.md"), "w" , encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logging.info("Evaluation complete. Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()

