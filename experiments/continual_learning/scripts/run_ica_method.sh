#!/bin/bash
#
# Run ICA-based continual learning experiments.
# Templates are built automatically if not found.
# Supports resume: safe to re-run after interruption.
#
# Usage: ./run_ica_method.sh [model] [mask_mode] [seeds]
#

set -e

MODEL=${1:-"llama-3.2-1b"}
MASK_MODE=${2:-"lesion"}
SEEDS=${3:-1}
OUTPUT_DIR="./experiments/continual_learning/results"

echo "=============================================="
echo "ICA-based Continual Learning Experiments"
echo "Model:     $MODEL"
echo "Mask Mode: $MASK_MODE"
echo "Seeds:     $SEEDS"
echo "=============================================="

poetry run fnsft-cl-orchestrate \
    --model "$MODEL" \
    --methods ica_networks \
    --output_dir "$OUTPUT_DIR" \
    --seeds "$SEEDS" \
    --ica_mask_mode "$MASK_MODE" \
    --phase all

echo ""
echo "=============================================="
echo "ICA experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
