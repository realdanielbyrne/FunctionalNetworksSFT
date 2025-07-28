#!/usr/bin/env python3
"""
Training Mode Demonstration Script

This script demonstrates how to use both PEFT and full parameter fine-tuning
modes in the FunctionalNetworksSFT framework using two different configuration approaches:

1. CLI Parameters: All configuration passed via command line arguments
2. YAML Config: Configuration loaded from YAML files with optional CLI overrides

Usage Examples:
    # Run all demonstrations (both modes, both methods)
    python examples/training_mode_demo.py

    # Run only PEFT training with both configuration methods
    python examples/training_mode_demo.py --mode peft

    # Run only CLI parameter demonstrations
    python examples/training_mode_demo.py --method cli

    # Run full fine-tuning with YAML config only
    python examples/training_mode_demo.py --mode full --method yaml

    # Clean up outputs after completion
    python examples/training_mode_demo.py --cleanup
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functionalnetworkssft.fnsft_trainer import main as train_main

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_peft_training_cli():
    """Demonstrate PEFT training using CLI parameters."""
    logger.info("🚀 Starting PEFT Training Demonstration (CLI Parameters)")

    # Simulate command line arguments for PEFT training
    sys.argv = [
        "fnsft_trainer.py",
        "--model_name_or_path",
        "microsoft/DialoGPT-small",  # Small model for demo
        "--dataset_name_or_path",
        "tatsu-lab/alpaca",
        "--output_dir",
        "./demo_output/peft_model_cli",
        # PEFT is enabled by default, no flag needed
        "--num_train_epochs",
        "1",  # Quick demo
        "--per_device_train_batch_size",
        "2",
        "--max_seq_length",
        "512",  # Shorter for demo
        "--learning_rate",
        "2e-4",
        "--logging_steps",
        "5",
        "--save_steps",
        "50",
        "--eval_steps",
        "50",
        "--validation_split",
        "0.1",
        "--use_4bit",  # Enable quantization
        "--lora_r",
        "8",  # Smaller rank for demo
        "--lora_alpha",
        "16",
    ]

    try:
        train_main()
        logger.info("✅ PEFT training (CLI) completed successfully!")
    except Exception as e:
        logger.error(f"❌ PEFT training (CLI) failed: {e}")
        return False

    return True


def run_peft_training_yaml():
    """Demonstrate PEFT training using YAML configuration."""
    logger.info("🚀 Starting PEFT Training Demonstration (YAML Config)")

    # Simulate command line arguments using YAML config
    sys.argv = [
        "fnsft_trainer.py",
        "--config",
        "examples/peft_training_config.yaml",
        "--output_dir",
        "./demo_output/peft_model_yaml",  # Override output dir
        "--num_train_epochs",
        "1",  # Override for quick demo
        "--max_seq_length",
        "512",  # Override for demo
        "--per_device_train_batch_size",
        "2",  # Override for demo
        "--logging_steps",
        "5",  # Override for demo
        "--save_steps",
        "50",  # Override for demo
        "--eval_steps",
        "50",  # Override for demo
    ]

    try:
        train_main()
        logger.info("✅ PEFT training (YAML) completed successfully!")
    except Exception as e:
        logger.error(f"❌ PEFT training (YAML) failed: {e}")
        return False

    return True


def run_full_training_cli():
    """Demonstrate full parameter fine-tuning using CLI parameters."""
    logger.info("🚀 Starting Full Fine-Tuning Demonstration (CLI Parameters)")

    # Simulate command line arguments for full fine-tuning
    sys.argv = [
        "fnsft_trainer.py",
        "--model_name_or_path",
        "microsoft/DialoGPT-small",  # Small model for demo
        "--dataset_name_or_path",
        "tatsu-lab/alpaca",
        "--output_dir",
        "./demo_output/full_model_cli",
        "--no_peft",  # Disable PEFT (use full fine-tuning)
        "--num_train_epochs",
        "1",  # Quick demo
        "--per_device_train_batch_size",
        "1",  # Smaller batch for full training
        "--gradient_accumulation_steps",
        "2",  # Maintain effective batch size
        "--max_seq_length",
        "512",  # Shorter for demo
        "--learning_rate",
        "5e-5",  # Lower learning rate for full training
        "--logging_steps",
        "5",
        "--save_steps",
        "50",
        "--eval_steps",
        "50",
        "--validation_split",
        "0.1",
        "--warmup_ratio",
        "0.1",  # More warmup for stability
        "--weight_decay",
        "0.01",  # Regularization
        "--gradient_checkpointing",  # Memory efficiency
    ]

    try:
        train_main()
        logger.info("✅ Full fine-tuning (CLI) completed successfully!")
    except Exception as e:
        logger.error(f"❌ Full fine-tuning (CLI) failed: {e}")
        return False

    return True


def run_full_training_yaml():
    """Demonstrate full parameter fine-tuning using YAML configuration."""
    logger.info("🚀 Starting Full Fine-Tuning Demonstration (YAML Config)")

    # Simulate command line arguments using YAML config
    sys.argv = [
        "fnsft_trainer.py",
        "--config",
        "examples/full_finetuning_config.yaml",
        "--output_dir",
        "./demo_output/full_model_yaml",  # Override output dir
        "--num_train_epochs",
        "1",  # Override for quick demo
        "--max_seq_length",
        "512",  # Override for demo
        "--per_device_train_batch_size",
        "1",  # Override for demo
        "--logging_steps",
        "5",  # Override for demo
        "--save_steps",
        "50",  # Override for demo
        "--eval_steps",
        "50",  # Override for demo
    ]

    try:
        train_main()
        logger.info("✅ Full fine-tuning (YAML) completed successfully!")
    except Exception as e:
        logger.error(f"❌ Full fine-tuning (YAML) failed: {e}")
        return False

    return True


def compare_outputs():
    """Compare the outputs of both training modes and configuration methods."""
    logger.info("\n" + "="*60)
    logger.info("📊 COMPARING TRAINING OUTPUTS")
    logger.info("="*60)

    # Define all possible output directories
    output_dirs = {
        "PEFT (CLI)": Path("./demo_output/peft_model_cli/final_model"),
        "PEFT (YAML)": Path("./demo_output/peft_model_yaml/final_model"),
        "Full (CLI)": Path("./demo_output/full_model_cli/final_model"),
        "Full (YAML)": Path("./demo_output/full_model_yaml/final_model"),
    }

    # Collect information about each output
    results = {}
    for name, dir_path in output_dirs.items():
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            size = sum(f.stat().st_size for f in files if f.is_file())

            # Check for specific file types
            has_adapter_config = (dir_path / "adapter_config.json").exists()
            has_adapter_model = any(f.name.startswith("adapter_model") for f in files)
            has_full_model = any(f.name.startswith("pytorch_model") or f.name.endswith(".safetensors") for f in files)

            results[name] = {
                "size": size,
                "files": [f.name for f in files],
                "adapter_config": has_adapter_config,
                "adapter_model": has_adapter_model,
                "full_model": has_full_model,
            }

    # Display results
    if results:
        logger.info("� Model Output Summary:")
        logger.info("-" * 60)

        for name, info in results.items():
            logger.info(f"\n🔹 {name}:")
            logger.info(f"   📁 Size: {info['size'] / 1024**2:.1f} MB")
            logger.info(f"   📄 Files: {len(info['files'])} files")
            logger.info(f"   � Has adapter config: {info['adapter_config']}")
            logger.info(f"   🔧 Has adapter weights: {info['adapter_model']}")
            logger.info(f"   🔧 Has full model: {info['full_model']}")

        # Compare sizes between PEFT and Full models
        peft_sizes = [info['size'] for name, info in results.items() if 'PEFT' in name]
        full_sizes = [info['size'] for name, info in results.items() if 'Full' in name]

        if peft_sizes and full_sizes:
            avg_peft_size = sum(peft_sizes) / len(peft_sizes)
            avg_full_size = sum(full_sizes) / len(full_sizes)
            ratio = avg_full_size / avg_peft_size if avg_peft_size > 0 else 0

            logger.info(f"\n💾 Size Comparison:")
            logger.info(f"   Average PEFT model size: {avg_peft_size / 1024**2:.1f} MB")
            logger.info(f"   Average Full model size: {avg_full_size / 1024**2:.1f} MB")
            logger.info(f"   Full models are {ratio:.1f}x larger than PEFT models")

        logger.info("\n✨ Key Observations:")
        logger.info("   • PEFT models contain adapter files (adapter_config.json, adapter_model.*)")
        logger.info("   • Full models contain complete model weights (pytorch_model.*, *.safetensors)")
        logger.info("   • Both CLI and YAML methods produce identical model structures")
        logger.info("   • PEFT models are significantly smaller and more portable")
    else:
        logger.warning("⚠️  No output directories found for comparison")
        logger.info("💡 Run the training demonstrations first to generate outputs")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Training Mode Demonstration")
    parser.add_argument(
        "--mode",
        choices=["peft", "full", "both"],
        default="both",
        help="Which training mode to demonstrate",
    )
    parser.add_argument(
        "--method",
        choices=["cli", "yaml", "both"],
        default="both",
        help="Which configuration method to demonstrate (CLI parameters vs YAML config)",
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Clean up demo outputs after completion"
    )

    args = parser.parse_args()

    logger.info("🎯 FunctionalNetworksSFT Training Mode Demonstration")
    logger.info("=" * 60)
    logger.info(f"📋 Training Mode: {args.mode}")
    logger.info(f"⚙️  Configuration Method: {args.method}")
    logger.info("=" * 60)

    success = True

    if args.mode in ["peft", "both"]:
        if args.method in ["cli", "both"]:
            logger.info("\n" + "=" * 50)
            logger.info("PEFT TRAINING - CLI PARAMETERS")
            logger.info("=" * 50)
            success &= run_peft_training_cli()

        if args.method in ["yaml", "both"]:
            logger.info("\n" + "=" * 50)
            logger.info("PEFT TRAINING - YAML CONFIG")
            logger.info("=" * 50)
            success &= run_peft_training_yaml()

    if args.mode in ["full", "both"]:
        if args.method in ["cli", "both"]:
            logger.info("\n" + "=" * 50)
            logger.info("FULL FINE-TUNING - CLI PARAMETERS")
            logger.info("=" * 50)
            success &= run_full_training_cli()

        if args.method in ["yaml", "both"]:
            logger.info("\n" + "=" * 50)
            logger.info("FULL FINE-TUNING - YAML CONFIG")
            logger.info("=" * 50)
            success &= run_full_training_yaml()

    # Always show comparison if any models were generated
    if success:
        compare_outputs()

    if args.cleanup and Path("./demo_output").exists():
        import shutil

        shutil.rmtree("./demo_output")
        logger.info("🧹 Cleaned up demo outputs")

    if success:
        logger.info("\n" + "="*60)
        logger.info("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("💡 Key Takeaways:")
        logger.info("   • CLI parameters: Direct control, good for experimentation")
        logger.info("   • YAML configs: Reproducible, version-controllable, good for production")
        logger.info("   • PEFT: Memory efficient, faster training, smaller outputs")
        logger.info("   • Full fine-tuning: Maximum performance, larger resource requirements")
        logger.info("\n� Next Steps:")
        logger.info("   • Check generated models in ./demo_output/ (if not cleaned up)")
        logger.info("   • Review examples/peft_training_config.yaml for PEFT setup")
        logger.info("   • Review examples/full_finetuning_config.yaml for full training setup")
        logger.info("   • See PEFT_VS_FULL_FINETUNING.md for detailed guidance")
    else:
        logger.error("💥 Demonstration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
