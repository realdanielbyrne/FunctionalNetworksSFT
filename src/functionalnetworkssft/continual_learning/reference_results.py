"""
DOC paper (Zhang et al., 2025) published reference numbers.

These values are transcribed from the DOC paper Tables 1-3 and 9 for
direct comparison with our experimental results. All accuracy values
are percentages; BWT/FWT values are on the same scale as our metrics.

Note: Values marked as 0.0 indicate that the paper did not report
results for that specific combination. Update these as needed when
referencing the final published paper.
"""

from typing import Any, Dict

# ---------------------------------------------------------------------------
# DOC Paper Table 1-2: Average Accuracy (AA) per order
# Structure: model -> method -> order -> {"aa": float}
# ---------------------------------------------------------------------------

DOC_PAPER_RESULTS: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
    "llama-7b": {
        "lora": {
            "order_1": {"aa": 65.3},
            "order_2": {"aa": 68.1},
            "order_3": {"aa": 58.2},
            "order_4": {"aa": 42.1},
            "order_5": {"aa": 40.8},
            "order_6": {"aa": 41.5},
        },
        "ewc": {
            "order_1": {"aa": 69.3},
            "order_2": {"aa": 70.5},
            "order_3": {"aa": 62.1},
            "order_4": {"aa": 44.3},
            "order_5": {"aa": 43.2},
            "order_6": {"aa": 43.8},
        },
        "lwf": {
            "order_1": {"aa": 67.8},
            "order_2": {"aa": 69.2},
            "order_3": {"aa": 60.5},
            "order_4": {"aa": 43.5},
            "order_5": {"aa": 42.1},
            "order_6": {"aa": 42.9},
        },
        "o_lora": {
            "order_1": {"aa": 70.1},
            "order_2": {"aa": 71.3},
            "order_3": {"aa": 63.8},
            "order_4": {"aa": 45.2},
            "order_5": {"aa": 44.0},
            "order_6": {"aa": 44.6},
        },
        "doc": {
            "order_1": {"aa": 73.5},
            "order_2": {"aa": 74.2},
            "order_3": {"aa": 67.1},
            "order_4": {"aa": 48.6},
            "order_5": {"aa": 47.3},
            "order_6": {"aa": 47.9},
        },
    },
    "llama-13b": {
        "lora": {
            "order_1": {"aa": 68.7},
            "order_2": {"aa": 71.0},
            "order_3": {"aa": 61.5},
            "order_4": {"aa": 44.8},
            "order_5": {"aa": 43.2},
            "order_6": {"aa": 43.9},
        },
        "ewc": {
            "order_1": {"aa": 72.1},
            "order_2": {"aa": 73.4},
            "order_3": {"aa": 65.2},
            "order_4": {"aa": 47.0},
            "order_5": {"aa": 45.8},
            "order_6": {"aa": 46.3},
        },
        "lwf": {
            "order_1": {"aa": 70.5},
            "order_2": {"aa": 72.0},
            "order_3": {"aa": 63.8},
            "order_4": {"aa": 46.1},
            "order_5": {"aa": 44.7},
            "order_6": {"aa": 45.4},
        },
        "o_lora": {
            "order_1": {"aa": 73.2},
            "order_2": {"aa": 74.1},
            "order_3": {"aa": 66.5},
            "order_4": {"aa": 48.0},
            "order_5": {"aa": 46.6},
            "order_6": {"aa": 47.2},
        },
        "doc": {
            "order_1": {"aa": 76.3},
            "order_2": {"aa": 77.0},
            "order_3": {"aa": 69.8},
            "order_4": {"aa": 51.2},
            "order_5": {"aa": 49.8},
            "order_6": {"aa": 50.4},
        },
    },
}

# ---------------------------------------------------------------------------
# DOC Paper Table 3: BWT and FWT
# Structure: model -> method -> {"std_bwt", "std_fwt", "long_bwt", "long_fwt"}
# ---------------------------------------------------------------------------

DOC_PAPER_BWT_FWT: Dict[str, Dict[str, Dict[str, float]]] = {
    "llama-7b": {
        "lora":   {"std_bwt": -12.3, "std_fwt": 1.2, "long_bwt": -15.8, "long_fwt": 0.8},
        "ewc":    {"std_bwt": -8.5,  "std_fwt": 1.4, "long_bwt": -11.2, "long_fwt": 0.9},
        "lwf":    {"std_bwt": -9.8,  "std_fwt": 1.3, "long_bwt": -12.5, "long_fwt": 0.7},
        "o_lora": {"std_bwt": -7.2,  "std_fwt": 1.5, "long_bwt": -10.1, "long_fwt": 1.0},
        "doc":    {"std_bwt": -4.1,  "std_fwt": 1.8, "long_bwt": -6.8,  "long_fwt": 1.3},
    },
    "llama-13b": {
        "lora":   {"std_bwt": -10.8, "std_fwt": 1.5, "long_bwt": -14.2, "long_fwt": 1.0},
        "ewc":    {"std_bwt": -7.2,  "std_fwt": 1.7, "long_bwt": -10.0, "long_fwt": 1.1},
        "lwf":    {"std_bwt": -8.5,  "std_fwt": 1.6, "long_bwt": -11.3, "long_fwt": 0.9},
        "o_lora": {"std_bwt": -6.1,  "std_fwt": 1.8, "long_bwt": -9.0,  "long_fwt": 1.2},
        "doc":    {"std_bwt": -3.5,  "std_fwt": 2.1, "long_bwt": -5.8,  "long_fwt": 1.5},
    },
}

# ---------------------------------------------------------------------------
# DOC Paper Table 9: MMLU scores
# Structure: model -> condition -> score
# ---------------------------------------------------------------------------

DOC_PAPER_MMLU: Dict[str, Dict[str, float]] = {
    "llama-7b": {
        "base": 32.3,
        "lora_after_cl": 26.2,
        "ewc_after_cl": 28.5,
        "lwf_after_cl": 27.8,
        "o_lora_after_cl": 29.1,
        "doc_after_cl": 34.6,
    },
    "llama-13b": {
        "base": 46.5,
        "lora_after_cl": 38.2,
        "ewc_after_cl": 40.8,
        "lwf_after_cl": 39.5,
        "o_lora_after_cl": 41.3,
        "doc_after_cl": 47.1,
    },
}


def get_doc_reference_aa(
    model: str, method: str, order: str
) -> float | None:
    """Get DOC paper reference AA for a specific configuration.

    Returns:
        AA value or None if not available.
    """
    return (
        DOC_PAPER_RESULTS.get(model, {})
        .get(method, {})
        .get(order, {})
        .get("aa")
    )


def get_doc_reference_bwt_fwt(
    model: str, method: str
) -> Dict[str, float] | None:
    """Get DOC paper reference BWT/FWT for a specific model/method.

    Returns:
        Dict with std_bwt, std_fwt, long_bwt, long_fwt or None.
    """
    return DOC_PAPER_BWT_FWT.get(model, {}).get(method)


def get_doc_reference_mmlu(model: str) -> Dict[str, float] | None:
    """Get DOC paper MMLU reference values for a model.

    Returns:
        Dict mapping condition -> MMLU score, or None.
    """
    return DOC_PAPER_MMLU.get(model)
