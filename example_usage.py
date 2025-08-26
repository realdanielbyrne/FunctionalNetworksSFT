#!/usr/bin/env python3
"""
Example usage of the build_ica_templates.py script

This script demonstrates how to create sample datasets and use the ICA template building script.
"""

import csv
import json
import os
import tempfile
import subprocess
import sys


def create_sample_datasets():
    """Create sample datasets for demonstration."""

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Creating sample datasets in: {temp_dir}")

    # Create CSV dataset
    csv_path = os.path.join(temp_dir, "science_qa.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["instruction", "response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        science_data = [
            {
                "instruction": "What is photosynthesis?",
                "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
            },
            {
                "instruction": "Explain Newton's first law of motion.",
                "response": "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force.",
            },
            {
                "instruction": "What is DNA?",
                "response": "DNA (Deoxyribonucleic acid) is a molecule that carries genetic instructions for the development, functioning, growth and reproduction of all known living organisms.",
            },
            {
                "instruction": "How does gravity work?",
                "response": "Gravity is a fundamental force that causes objects with mass to attract each other. The strength of gravitational attraction depends on the masses of the objects and the distance between them.",
            },
            {
                "instruction": "What is the water cycle?",
                "response": "The water cycle is the continuous movement of water through evaporation, condensation, precipitation, and collection, cycling water through Earth's atmosphere, land, and oceans.",
            },
        ]

        # Repeat data to have more samples
        for i in range(20):
            for item in science_data:
                writer.writerow(
                    {
                        "instruction": f"{item['instruction']} (Sample {i})",
                        "response": f"{item['response']} (Response {i})",
                    }
                )

    print(f"Created CSV dataset: {csv_path}")

    # Create JSON dataset
    json_path = os.path.join(temp_dir, "general_qa.json")
    general_data = []
    base_qa = [
        {
            "instruction": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        },
        {
            "instruction": "How do neural networks work?",
            "response": "Neural networks are computing systems inspired by biological neural networks, using interconnected nodes (neurons) to process information and learn patterns from data.",
        },
        {
            "instruction": "What is climate change?",
            "response": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities that increase greenhouse gas emissions.",
        },
        {
            "instruction": "Explain renewable energy.",
            "response": "Renewable energy comes from natural sources that are constantly replenished, such as solar, wind, hydroelectric, and geothermal power, providing sustainable alternatives to fossil fuels.",
        },
        {
            "instruction": "What is quantum computing?",
            "response": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that could solve certain problems much faster than classical computers.",
        },
    ]

    # Repeat data to have more samples
    for i in range(15):
        for item in base_qa:
            general_data.append(
                {
                    "instruction": f"{item['instruction']} (Sample {i})",
                    "response": f"{item['response']} (Response {i})",
                }
            )

    with open(json_path, "w") as f:
        json.dump(general_data, f, indent=2)

    print(f"Created JSON dataset: {json_path}")

    return temp_dir, csv_path, json_path


def run_ica_template_building(csv_path, json_path, output_dir):
    """Run the ICA template building script."""

    print("\nRunning ICA template building...")
    print(
        "Note: This will download a model if not already cached, which may take some time."
    )

    cmd = [
        "python",
        "-m",
        "functionalnetworkssft.build_ica_templates",
        "microsoft/DialoGPT-small",  # Model as first positional argument
        csv_path,
        json_path,
        "--ica_template_samples_per_ds",
        "20",
        "--ica_template_output",
        output_dir,
        "--ica_components",
        "5",
        "--ica_percentile",
        "95.0",
        "--ica_dtype",
        "float32",
        "--max_seq_length",
        "256",
        "--template_format",
        "auto",
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ ICA template building completed successfully!")
            print("\nOutput:")
            print(result.stdout)

            # Check if templates were created
            template_file = os.path.join(output_dir, "global_templates.json")
            if os.path.exists(template_file):
                print(f"\n📁 Templates saved to: {template_file}")

                # Load and display template info
                with open(template_file, "r") as f:
                    templates = json.load(f)

                print(f"Template name: {templates['name']}")
                print(f"Number of components: {len(templates['templates'])}")
                print(f"Hidden size: {templates['layout']['hidden_size']}")
                print(
                    f"Captured layers: {len(templates['layout']['captured_layers_sorted'])}"
                )

                # Show component details
                for comp_id, comp_data in templates["templates"].items():
                    layer_count = len(comp_data)
                    total_channels = sum(
                        len(channels) for channels in comp_data.values()
                    )
                    print(
                        f"Component {comp_id}: {total_channels} channels across {layer_count} layers"
                    )

                return True
            else:
                print("❌ Template file not found!")
                return False
        else:
            print("❌ ICA template building failed!")
            print("Error output:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ Command timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def main():
    """Main demonstration function."""

    print("ICA Template Building Demonstration")
    print("=" * 50)

    # Ask user if they want to run the demo
    response = (
        input(
            "This demo will download a model and may take several minutes. Continue? (y/N): "
        )
        .strip()
        .lower()
    )

    if response not in ["y", "yes"]:
        print("Demo cancelled.")
        return

    try:
        # Create sample datasets
        temp_dir, csv_path, json_path = create_sample_datasets()

        # Create output directory
        output_dir = os.path.join(temp_dir, "templates")
        os.makedirs(output_dir, exist_ok=True)

        # Run ICA template building
        success = run_ica_template_building(csv_path, json_path, output_dir)

        if success:
            print("\n🎉 Demo completed successfully!")
            print(f"All files are in: {temp_dir}")
            print("\nYou can now use the generated templates in training with:")
            print(
                f"--ica_template_path {os.path.join(output_dir, 'global_templates.json')}"
            )
        else:
            print("\n❌ Demo failed!")

        # Ask if user wants to clean up
        cleanup = (
            input(f"\nDelete temporary files in {temp_dir}? (y/N): ").strip().lower()
        )
        if cleanup in ["y", "yes"]:
            import shutil

            shutil.rmtree(temp_dir)
            print("Temporary files cleaned up.")
        else:
            print(f"Temporary files kept in: {temp_dir}")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")


if __name__ == "__main__":
    main()
