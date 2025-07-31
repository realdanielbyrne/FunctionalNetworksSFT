#!/usr/bin/env python3
"""
Llama-3.2-1B Sarcasm Fine-Tuning Example

This script demonstrates fine-tuning the meta-llama/Llama-3.2-1B-Instruct model
on a sarcasm detection dataset using both PEFT (LoRA) and full parameter fine-tuning modes.

The example showcases:
1. PEFT training with LoRA adapters (memory-efficient)
2. Full parameter fine-tuning (higher memory requirements)
3. Proper configuration for non-quantized models
4. Chat template handling for instruction-following models
5. Dataset preprocessing for question-answer format

Dataset: sarcasm.csv - Contains question-answer pairs with sarcastic responses
Model: meta-llama/Llama-3.2-1B-Instruct (non-quantized)

Usage:
    # Run both PEFT and full training demonstrations
    python examples/llama_sarcasm_example.py

    # Run only PEFT training
    python examples/llama_sarcasm_example.py --mode peft

    # Run only full fine-tuning
    python examples/llama_sarcasm_example.py --mode full

    # Clean up outputs after completion
    python examples/llama_sarcasm_example.py --cleanup
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Set MPS environment variables for better compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the src directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.fnsft_trainer import main as train_main

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_peft_training():
    """Demonstrate PEFT training with Llama-3.2-1B-Instruct on sarcasm dataset."""
    logger.info("🚀 Starting PEFT Training with Llama-3.2-1B-Instruct")
    logger.info("📊 Dataset: Sarcasm Question-Answer Pairs")
    logger.info("🔧 Mode: Parameter-Efficient Fine-Tuning (LoRA)")
    logger.info("💾 Memory: Optimized for efficiency")

    # Simulate command line arguments for PEFT training
    sys.argv = [
        "fnsft_trainer.py",
        "--model_name_or_path",
        "meta-llama/Llama-3.2-1B-Instruct",  # Non-quantized Llama model
        "--torch_dtype",
        "float32",  # Use float32 for MPS compatibility (no mixed precision issues)
        "--dataset_name_or_path",
        "datasets/sarcasm.csv",  # Local sarcasm dataset
        "--output_dir",
        "./demo_output/llama_sarcasm_peft",
        # PEFT Configuration (enabled by default)
        "--lora_r",
        "16",  # LoRA rank
        "--lora_alpha",
        "32",  # LoRA alpha scaling
        "--lora_dropout",
        "0.1",  # LoRA dropout
        # Training Configuration
        "--num_train_epochs",
        "1",  # Quick demo for debugging
        "--per_device_train_batch_size",
        "4",  # Batch size for PEFT
        "--per_device_eval_batch_size",
        "4",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "2e-4",  # Higher learning rate for PEFT
        "--weight_decay",
        "0.001",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        # Sequence and Template Configuration
        "--max_seq_length",
        "1024",  # Reasonable length for Q&A
        "--template_format",
        "auto",  # Auto-detect chat template
        "--validation_split",
        "0.1",  # 10% for validation
        # Memory Optimization (gradient_checkpointing is enabled by default)
        # Quantization (disabled for non-quantized model - don't include flags to disable)
        # Logging and Evaluation
        "--logging_steps",
        "10",
        "--save_steps",
        "100",
        "--eval_steps",
        "100",
        "--save_total_limit",
        "2",
        "--load_best_model_at_end",
        "--metric_for_best_model",
        "eval_loss",
        # Authentication (if needed for gated model)
        "--use_auth_token",
    ]

    try:
        train_main()
        logger.info("✅ PEFT training completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ PEFT training failed: {e}")
        return False


def run_full_training():
    """Demonstrate full parameter fine-tuning with Llama-3.2-1B-Instruct."""
    logger.info("🚀 Starting Full Fine-Tuning with Llama-3.2-1B-Instruct")
    logger.info("📊 Dataset: Sarcasm Question-Answer Pairs")
    logger.info("🔧 Mode: Full Parameter Fine-Tuning")
    logger.info("💾 Memory: Higher requirements, better performance")

    # Simulate command line arguments for full fine-tuning
    sys.argv = [
        "fnsft_trainer.py",
        "--model_name_or_path",
        "meta-llama/Llama-3.2-1B-Instruct",  # Non-quantized Llama model
        "--torch_dtype",
        "float32",  # Use float32 for MPS compatibility (no mixed precision issues)
        "--dataset_name_or_path",
        "datasets/sarcasm.csv",  # Local sarcasm dataset
        "--output_dir",
        "./demo_output/llama_sarcasm_full",
        # Full Fine-Tuning Configuration
        "--no_peft",  # Disable PEFT for full training
        # Training Configuration (conservative for stability)
        "--num_train_epochs",
        "3",  # Fewer epochs for full training demo
        "--per_device_train_batch_size",
        "2",  # Smaller batch size for full training
        "--per_device_eval_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "2",  # Maintain effective batch size
        "--learning_rate",
        "5e-5",  # Lower learning rate for full training
        "--weight_decay",
        "0.01",  # Higher weight decay for regularization
        "--warmup_ratio",
        "0.1",  # More warmup for stability
        "--lr_scheduler_type",
        "linear",  # Linear scheduler for full training
        # Sequence and Template Configuration
        "--max_seq_length",
        "1024",  # Same as PEFT for comparison
        "--template_format",
        "auto",  # Auto-detect chat template
        "--validation_split",
        "0.1",  # 10% for validation
        # Memory Optimization (gradient_checkpointing is enabled by default)
        # Quantization (disabled for non-quantized model - don't include flags to disable)
        # Logging and Evaluation
        "--logging_steps",
        "5",  # More frequent logging
        "--save_steps",
        "50",
        "--eval_steps",
        "50",
        "--save_total_limit",
        "2",
        "--load_best_model_at_end",
        "--metric_for_best_model",
        "eval_loss",
        # Authentication (if needed for gated model)
        "--use_auth_token",
        # Debug mode to help diagnose gradient issues
        "--logging_steps",
        "1",  # More frequent logging for debugging
    ]

    try:
        train_main()
        logger.info("✅ Full fine-tuning completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Full fine-tuning failed: {e}")
        return False


def compare_outputs():
    """Compare the outputs from both training modes."""
    logger.info("\n" + "=" * 60)
    logger.info("📊 TRAINING COMPARISON SUMMARY")
    logger.info("=" * 60)

    peft_dir = Path("./demo_output/llama_sarcasm_peft")
    full_dir = Path("./demo_output/llama_sarcasm_full")

    if peft_dir.exists():
        logger.info(f"📁 PEFT Model: {peft_dir}")
        logger.info("   - Contains LoRA adapters only")
        logger.info("   - Small file size (~few MB)")
        logger.info("   - Memory efficient training")
        logger.info("   - Requires base model + adapters for inference")

    if full_dir.exists():
        logger.info(f"📁 Full Model: {full_dir}")
        logger.info("   - Contains complete fine-tuned model")
        logger.info("   - Large file size (~few GB)")
        logger.info("   - Higher memory training requirements")
        logger.info("   - Self-contained for inference")

    logger.info("\n💡 Key Differences:")
    logger.info("   • PEFT: Fast, memory-efficient, requires base model")
    logger.info("   • Full: Slower, memory-intensive, standalone model")
    logger.info("   • Both: Trained on sarcasm Q&A for instruction following")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Llama-3.2-1B Sarcasm Fine-Tuning Demonstration"
    )
    parser.add_argument(
        "--mode",
        choices=["peft", "full", "both"],
        default="both",
        help="Which training mode to demonstrate (default: both)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up demo outputs after completion",
    )

    args = parser.parse_args()

    logger.info("🎯 Llama-3.2-1B Sarcasm Fine-Tuning Demonstration")
    logger.info("=" * 60)
    logger.info(f"🤖 Model: meta-llama/Llama-3.2-1B-Instruct")
    logger.info(f"📊 Dataset: datasets/sarcasm.csv (200 Q&A pairs)")
    logger.info(f"🔧 Training Mode: {args.mode}")
    logger.info("=" * 60)

    success = True

    if args.mode in ["peft", "both"]:
        logger.info("\n" + "=" * 50)
        logger.info("PEFT TRAINING - LLAMA SARCASM")
        logger.info("=" * 50)
        success &= run_peft_training()

    if args.mode in ["full", "both"]:
        logger.info("\n" + "=" * 50)
        logger.info("FULL FINE-TUNING - LLAMA SARCASM")
        logger.info("=" * 50)
        success &= run_full_training()

    # Show comparison if any models were generated
    if success:
        compare_outputs()

    if args.cleanup and Path("./demo_output").exists():
        import shutil

        shutil.rmtree("./demo_output")
        logger.info("🧹 Cleaned up demo outputs")

    if success:
        logger.info("\n🎉 Demonstration completed successfully!")
        logger.info("💡 Try running inference with your fine-tuned models!")
    else:
        logger.error("\n❌ Some demonstrations failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
