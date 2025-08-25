# ICA Template Building Script

This repository contains a command-line script for building ICA (Independent Component Analysis) templates from datasets without requiring model training. The generated templates can be used later during training to apply pre-computed functional network masks.

## Overview

The `build_ica_templates.py` script implements the following workflow:

1. **Load Datasets**: Accepts multiple dataset paths (CSV, JSON, JSONL formats)
2. **Sample Data**: Extracts a specified number of samples from each dataset
3. **Combine Data**: Stacks all sampled data into a unified dataset
4. **Compute ICA**: Runs ICA analysis on the combined dataset using a specified model
5. **Build Templates**: Creates templates from the computed ICA components
6. **Save Results**: Saves templates as `global_templates.json` for later use

## Installation

Install the package with:

```bash
pip install -e .
```

This will install all required dependencies and make the `build-ica-templates` command available.

## Usage

### Basic Command

```bash
build-ica-templates \
    --ica_build_templates_from dataset1.csv dataset2.json \
    --ica_template_samples_per_ds 100 \
    --ica_template_output ./output/templates/ \
    --model_name_or_path microsoft/DialoGPT-medium
```

### Alternative: Direct Python Module

```bash
python -m functionalnetworkssft.build_ica_templates \
    --ica_build_templates_from dataset1.csv dataset2.json \
    --ica_template_samples_per_ds 100 \
    --ica_template_output ./output/templates/ \
    --model_name_or_path microsoft/DialoGPT-medium
```

### Required Arguments

- `--ica_build_templates_from`: One or more dataset paths to build templates from
- `--ica_template_samples_per_ds`: Number of samples to extract from each dataset
- `--ica_template_output`: Output directory for saving the generated templates
- `--model_name_or_path`: Model name or path to use for ICA computation

### Optional Arguments

- `--ica_components`: Number of ICA components to extract (default: 10)
- `--ica_percentile`: Percentile threshold for component selection (default: 98.0)
- `--ica_dtype`: Data type for ICA computation (default: auto)
- `--max_seq_length`: Maximum sequence length for tokenization (default: 512)
- `--template_format`: Dataset format detection mode (default: auto)

### Example with All Options

```bash
build-ica-templates \
    --ica_build_templates_from science_qa.csv general_qa.json \
    --ica_template_samples_per_ds 50 \
    --ica_template_output ./templates/ \
    --model_name_or_path microsoft/DialoGPT-small \
    --ica_components 5 \
    --ica_percentile 95.0 \
    --ica_dtype float32 \
    --max_seq_length 256 \
    --template_format auto
```

## Supported Dataset Formats

The script supports multiple dataset formats:

### CSV Format

```csv
instruction,response
"What is machine learning?","Machine learning is a subset of AI..."
"Explain photosynthesis","Photosynthesis is the process..."
```

### JSON Format

```json
[
  {
    "instruction": "What is machine learning?",
    "response": "Machine learning is a subset of AI..."
  },
  {
    "instruction": "Explain photosynthesis", 
    "response": "Photosynthesis is the process..."
  }
]
```

### JSONL Format

```jsonl
{"instruction": "What is machine learning?", "response": "Machine learning is a subset of AI..."}
{"instruction": "Explain photosynthesis", "response": "Photosynthesis is the process..."}
```

## Output

The script generates a `global_templates.json` file containing:

- **Template metadata**: Name, layout information
- **Component data**: Per-component channel mappings across layers
- **Layout info**: Captured layers and hidden size

Example output structure:

```json
{
  "name": "global_templates_v1",
  "layout": {
    "captured_layers_sorted": [0, 1, 2, ...],
    "hidden_size": 768
  },
  "templates": {
    "0": {
      "0": [1, 2, 3, ...],
      "1": [4, 5, 6, ...]
    },
    "1": {
      "0": [7, 8, 9, ...],
      "1": [10, 11, 12, ...]
    }
  }
}
```

## Using Templates in Training

Once generated, use the templates in training with:

```bash
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name_or_path your_training_data.json \
    --output_dir ./output \
    --mask_mode key \
    --ica_template_path ./templates/global_templates.json \
    --ica_component_ids 0 1 2
```

## Testing

Run the unit tests:

```bash
python test_build_ica_templates.py
```

Run integration tests (requires model download):

```bash
python test_build_ica_templates.py --integration
```

## Example Demo

Run the example demonstration:

```bash
python example_usage.py
```

This will:

1. Create sample datasets
2. Run the ICA template building process
3. Show the results
4. Optionally clean up temporary files

## Technical Details

### ICA Computation

- Uses FastICA algorithm for component analysis
- Operates on global feature space (concatenated MLP outputs)
- Supports configurable number of components and percentile thresholds

### Memory Optimization

- Supports different data types (float32, float16, bfloat16)
- Limits sample size for ICA computation (max 1024 samples)
- Uses efficient dataset sampling and loading

### Error Handling

- Validates dataset paths and arguments
- Handles missing or corrupted datasets gracefully
- Provides detailed logging and progress indicators

## Performance Considerations

- **Model Size**: Smaller models (e.g., DialoGPT-small) are faster for template building
- **Sample Count**: More samples provide better ICA analysis but increase computation time
- **Sequence Length**: Shorter sequences reduce memory usage and computation time
- **Data Type**: float16/bfloat16 can improve performance with minimal accuracy loss

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--max_seq_length` or `--ica_template_samples_per_ds`
2. **Model Download Fails**: Check internet connection and HuggingFace access
3. **Dataset Format Error**: Ensure datasets have 'instruction' and 'response' columns
4. **Permission Error**: Ensure write access to output directory

### Debug Mode

Add verbose logging by modifying the script's logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to this script:

1. Run unit tests before submitting changes
2. Update documentation for new features
3. Follow existing code style and patterns
4. Add tests for new functionality

## License

This script is part of the FunctionalNetworksSFT project and follows the same license terms.
