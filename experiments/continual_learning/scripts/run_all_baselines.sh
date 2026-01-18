#!/bin/bash
#
# Run all baseline methods on all task orders
# Usage: ./run_all_baselines.sh [model] [output_dir]
#

set -e

MODEL=${1:-"llama-3.2-1b"}
OUTPUT_DIR=${2:-"./experiments/continual_learning/results"}
SEED=42

# Methods to run
METHODS=("lora" "ewc" "lwf" "o_lora" "doc" "ica_networks")

# Task orders
STANDARD_ORDERS=("order_1" "order_2" "order_3")
LONG_ORDERS=("order_4" "order_5" "order_6")

echo "=============================================="
echo "Continual Learning Evaluation"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Standard CL Benchmark
echo ""
echo "=== Standard CL Benchmark ==="
for method in "${METHODS[@]}"; do
    for order in "${STANDARD_ORDERS[@]}"; do
        echo ""
        echo "Running $method on $order..."
        poetry run fnsft-cl-eval \
            --model $MODEL \
            --method $method \
            --task_order $order \
            --output_dir "$OUTPUT_DIR/standard" \
            --seed $SEED \
            --no_baselines
    done
done

# Long Chain of Tasks
echo ""
echo "=== Long Chain of Tasks ==="
for method in "${METHODS[@]}"; do
    for order in "${LONG_ORDERS[@]}"; do
        echo ""
        echo "Running $method on $order..."
        poetry run fnsft-cl-eval \
            --model $MODEL \
            --method $method \
            --task_order $order \
            --output_dir "$OUTPUT_DIR/long_chain" \
            --seed $SEED \
            --no_baselines
    done
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Generate comparison tables
echo ""
echo "Generating comparison tables..."
poetry run fnsft-cl-tables --results_dir "$OUTPUT_DIR" --model "$MODEL"

