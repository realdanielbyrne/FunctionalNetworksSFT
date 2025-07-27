# Intelligent Chat Template Handling

This document describes the new intelligent chat template handling feature implemented in the `InstructionDataset` class.

## Overview

The feature automatically detects and uses the appropriate chat template format for optimal fine-tuning performance, while providing flexible override options for specific use cases.

## Key Features

### 1. Auto-detection

- **Smart Detection**: Automatically checks if the loaded tokenizer has a `chat_template` attribute
- **Fallback Strategy**: Uses tokenizer's native chat template if available, otherwise falls back to basic template
- **Transparency**: Logs which template format is being used for clear visibility

### 2. CLI Override Options

`--template_format` argument with the following choices:

- **`auto`** (default): Use tokenizer's chat template if available, otherwise fall back to basic template
- **`chat`**: Force use of tokenizer's chat template (error if not available)
- **`alpaca`**: Use Alpaca format: `### Instruction:\n{instruction}\n\n### Response:\n{response}`
- **`chatml`**: Use ChatML format: `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>`
- **`basic`**: Use the current basic instruction template

### 3. Implementation Details

#### Modified Components

1. **`DataArguments`**: Added `template_format` field
2. **`InstructionDataset.__init__()`**: Added `template_format` parameter and auto-detection logic
3. **`InstructionDataset.__getitem__()`**: Updated to apply appropriate template based on format
4. **Argument Parser**: Added `--template_format` CLI option
5. **Dataset Creation**: Updated calls to pass template format parameter

#### Template Handling

- **Chat Templates**: Convert instruction/response pairs to messages format: `[{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]`
- **Error Handling**: Proper validation and error messages for unsupported template formats
- **Type Safety**: Ensures chat template returns string format

## Usage Examples

### CLI Usage

```bash
# Auto-detect template format (default)
python -m src.functionalnetworkssft.fnsft_trainer \
  --model_name_or_path microsoft/DialoGPT-medium \
  --dataset_name_or_path data.json \
  --output_dir ./output \
  --template_format auto

# Force chat template usage
python -m src.functionalnetworkssft.fnsft_trainer \
  --model_name_or_path microsoft/DialoGPT-medium \
  --dataset_name_or_path data.json \
  --output_dir ./output \
  --template_format chat

# Use Alpaca format
python -m src.functionalnetworkssft.fnsft_trainer \
  --model_name_or_path gpt2 \
  --dataset_name_or_path data.json \
  --output_dir ./output \
  --template_format alpaca

# Use ChatML format
python -m src.functionalnetworkssft.fnsft_trainer \
  --model_name_or_path gpt2 \
  --dataset_name_or_path data.json \
  --output_dir ./output \
  --template_format chatml

# Use basic format with custom template
python -m src.functionalnetworkssft.fnsft_trainer \
  --model_name_or_path gpt2 \
  --dataset_name_or_path data.json \
  --output_dir ./output \
  --template_format basic \
  --instruction_template "Q: {instruction}\nA: {response}"
```

### Programmatic Usage

```python
from src.functionalnetworkssft.sft_trainer import InstructionDataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Create dataset with auto-detection
dataset = InstructionDataset(
    data=data,
    tokenizer=tokenizer,
    template_format="auto"  # Will auto-detect and use chat template if available
)

# Force specific format
dataset = InstructionDataset(
    data=data,
    tokenizer=tokenizer,
    template_format="alpaca"  # Force Alpaca format
)
```

## Template Format Examples

Given instruction: "What is the capital of France?" and response: "The capital of France is Paris."

### Auto-detected Chat Template

```text
user: What is the capital of France?
assistant: The capital of France is Paris.
```

### Alpaca Format

```text
### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris.
```

### ChatML Format

```text
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

### Basic Format (with custom template)

```text
Q: What is the capital of France?
A: The capital of France is Paris.
```

## Benefits

1. **Optimal Performance**: Uses the correct template format that matches what the model was originally trained on
2. **Flexibility**: Provides override options for specific use cases
3. **Backward Compatibility**: Existing workflows continue to work without changes
4. **Transparency**: Clear logging of which template format is being used
5. **Error Prevention**: Validates template availability and provides helpful error messages

## Testing

Comprehensive test suite includes:

- Auto-detection with and without chat templates
- Forced template format usage
- Error handling for invalid formats
- CLI argument parsing
- Template formatting validation

Run tests with:

```bash
python -m pytest tests/test_template_handling.py tests/test_template_cli_integration.py -v
```

## Demo

Run the demonstration script to see the feature in action:

```bash
python examples/template_format_demo.py
```

This feature ensures optimal fine-tuning performance by using the correct template format while maintaining flexibility for various use cases.
