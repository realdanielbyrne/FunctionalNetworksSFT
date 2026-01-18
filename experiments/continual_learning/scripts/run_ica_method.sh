#!/bin/bash
#
# Run ICA-based continual learning experiments
# Usage: ./run_ica_method.sh [model] [ica_template_path] [mask_mode]
#

set -e

MODEL=${1:-"llama-3.2-1b"}
ICA_TEMPLATE_PATH=${2:-""}
MASK_MODE=${3:-"lesion"}
OUTPUT_DIR="./experiments/continual_learning/results/ica_experiments"
SEED=42

echo "=============================================="
echo "ICA-based Continual Learning Experiments"
echo "Model: $MODEL"
echo "Mask Mode: $MASK_MODE"
echo "Template Path: $ICA_TEMPLATE_PATH"
echo "=============================================="

# Standard benchmark orders
ORDERS=("order_1" "order_2" "order_3")

for order in "${ORDERS[@]}"; do
    echo ""
    echo "Running ICA Networks on $order with $MASK_MODE mode..."
    
    CMD="poetry run fnsft-cl-eval \
        --model $MODEL \
        --method ica_networks \
        --task_order $order \
        --output_dir $OUTPUT_DIR \
        --seed $SEED \
        --ica_mask_mode $MASK_MODE \
        --ica_components 10 \
        --no_baselines"
    
    if [ -n "$ICA_TEMPLATE_PATH" ]; then
        CMD="$CMD --ica_template_path $ICA_TEMPLATE_PATH"
    fi
    
    eval $CMD
done

echo ""
echo "=============================================="
echo "ICA experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Generate tables
poetry run fnsft-cl-tables --results_dir "$OUTPUT_DIR" --model "$MODEL"

