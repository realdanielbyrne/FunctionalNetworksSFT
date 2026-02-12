#!/bin/bash
#
# Run all CL experiments with the orchestrator.
# Supports resume: safe to re-run after interruption.
#
# Usage: ./run_all_baselines.sh [model] [output_dir] [seeds]
#

set -e

MODEL=${1:-"llama-3.2-1b"}
OUTPUT_DIR=${2:-"./experiments/continual_learning/results"}
SEEDS=${3:-1}

echo "=============================================="
echo "Continual Learning Experiment Suite"
echo "Model:  $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Seeds:  $SEEDS"
echo "=============================================="

poetry run fnsft-cl-orchestrate \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --seeds "$SEEDS" \
    --phase all

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
