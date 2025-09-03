import argparse
import logging
import os
import sys

from .utils.model_utils import convert_to_gguf


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def resolve_model_dir(input_dir: str | None, model_path: str | None, select: str) -> tuple[str, str]:
    """
    Determine which directory to convert to GGUF.

    Returns (model_dir, root_dir_for_default_outfile)
    """
    if model_path:
        model_dir = os.path.abspath(model_path)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"--model-path does not exist or is not a directory: {model_dir}")
        return model_dir, os.path.dirname(model_dir)

    root = os.path.abspath(input_dir) if input_dir else os.getcwd()
    merged = os.path.join(root, "merged_model")
    final = os.path.join(root, "final_model")

    if select == "merged":
        if not os.path.isdir(merged):
            raise FileNotFoundError(f"merged_model not found under: {root}")
        return merged, root
    elif select == "final":
        if not os.path.isdir(final):
            raise FileNotFoundError(f"final_model not found under: {root}")
        return final, root
    else:  # auto
        if os.path.isdir(merged):
            return merged, root
        if os.path.isdir(final):
            return final, root
        raise FileNotFoundError(
            f"Neither merged_model nor final_model found under: {root}. "
            "Provide --model-path to a specific model directory or --input-dir with subfolders."
        )


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Convert a merged or final model directory to GGUF (via llama.cpp's converter).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing merged_model/ and/or final_model subdirectories (defaults to CWD)",
    )
    group.add_argument(
        "--model-path",
        type=str,
        help="Direct path to a model directory (e.g., .../merged_model or .../final_model)",
    )

    parser.add_argument(
        "--select",
        choices=["auto", "merged", "final"],
        default="auto",
        help="Which directory to convert when using --input-dir (default: auto prefers merged)",
    )
    parser.add_argument(
        "--quantization",
        "--outtype",
        dest="quantization",
        default="q4_0",
        help="GGUF quantization type (e.g., q4_0, q8_0, f16). Default: q4_0",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="Output GGUF filepath. Defaults to <root>/model.gguf where root is --input-dir or parent of --model-path.",
    )

    args = parser.parse_args(argv)

    try:
        model_dir, root_dir = resolve_model_dir(args.input_dir, args.model_path, args.select)
        outfile = args.outfile or os.path.join(root_dir, "model.gguf")

        logging.info(f"Converting model directory: {model_dir}")
        logging.info(f"Output GGUF path: {outfile}")
        logging.info(f"Quantization: {args.quantization}")

        convert_to_gguf(model_dir, outfile, args.quantization)
        logging.info("Conversion complete")
        return 0
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

