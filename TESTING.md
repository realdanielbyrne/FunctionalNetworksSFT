# Testing Guide for FunctionalNetworksSFT

This document describes how to run unit tests for the FunctionalNetworksSFT application. It is optimized for both human readability and LLM agent parsing.

## Quick Start

```bash
# Activate virtual environment
eval $(poetry env activate)

# Run all tests
poetry run pytest

# Run tests with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_ica_masking.py

# Run installation verification
python tests/test_installation.py
```

## Test Framework

- **Primary Framework**: pytest (>=7.0.0)
- **Secondary Framework**: unittest (for some test classes)
- **Test Directory**: `tests/`
- **Naming Convention**: `test_*.py`

## Test Structure

### Test Categories

1. **Unit Tests**
   - `test_ica_masking.py` - Core ICA masking functionality
   - `test_ica_mask_application.py` - Mask application and hooks
   - `test_template_handling.py` - Chat template processing

2. **Integration Tests**
   - `test_ica_cli_integration.py` - Command-line interface integration
   - `test_template_cli_integration.py` - Template CLI integration

3. **System Tests**
   - `test_installation.py` - Installation verification and platform detection
   - `test_ica_suite.py` - Comprehensive ICA functionality test suite

### Test File Organization

```
tests/
├── __init__.py
├── test_ica_masking.py           # Core ICA functionality
├── test_ica_mask_application.py  # Mask application logic
├── test_ica_cli_integration.py   # CLI integration tests
├── test_ica_suite.py            # Comprehensive ICA test suite
├── test_installation.py         # Installation verification
├── test_template_handling.py    # Template processing tests
└── test_template_cli_integration.py  # Template CLI tests
```

## Running Tests

### Prerequisites

```bash
# Ensure dependencies are installed
poetry install

# Activate virtual environment (recommended)
eval $(poetry env activate)
```

### Basic Test Commands

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/functionalnetworkssft

# Run specific test file
poetry run pytest tests/test_ica_masking.py

# Run specific test class
poetry run pytest tests/test_ica_masking.py::TestICAMasking

# Run specific test method
poetry run pytest tests/test_ica_masking.py::TestICAMasking::test_apply_ica_masks_basic

# Run tests matching pattern
poetry run pytest -k "ica_mask"

# Run tests with verbose output
poetry run pytest -v

# Run tests and stop on first failure
poetry run pytest -x

# Run tests in parallel (if pytest-xdist installed)
poetry run pytest -n auto
```

### Specialized Test Runners

```bash
# Installation verification (standalone script)
python tests/test_installation.py

# Comprehensive ICA test suite (standalone script)
python tests/test_ica_suite.py
```

### Test Output Levels

- **Minimal**: `poetry run pytest` (default)
- **Verbose**: `poetry run pytest -v` (shows individual test names)
- **Very Verbose**: `poetry run pytest -vv` (shows test names and docstrings)

## Test Configuration

### Environment Setup

Tests automatically handle:
- Project root path addition to `sys.path`
- Mock objects for external dependencies
- Temporary file creation and cleanup
- Cross-platform device detection

### Mock Objects

Tests use extensive mocking for:
- PyTorch models (`MockModel`, `MockLinearModule`)
- Tokenizers (`MockTokenizer`)
- Datasets (`MockDataset`)
- External libraries (FastICA, transformers)

## Test Categories by Functionality

### ICA Masking Tests

```bash
# Core ICA functionality
poetry run pytest tests/test_ica_masking.py

# Mask application and hooks
poetry run pytest tests/test_ica_mask_application.py

# CLI integration
poetry run pytest tests/test_ica_cli_integration.py

# Complete ICA test suite
python tests/test_ica_suite.py
```

### Template Handling Tests

```bash
# Template processing logic
poetry run pytest tests/test_template_handling.py

# Template CLI integration
poetry run pytest tests/test_template_cli_integration.py
```

### System Tests

```bash
# Installation and platform verification
python tests/test_installation.py
```

## Test Naming Conventions

### Test Files
- Format: `test_<module_name>.py`
- Examples: `test_ica_masking.py`, `test_template_handling.py`

### Test Classes
- Format: `Test<ClassName>`
- Examples: `TestICAMasking`, `TestInstructionDatasetTemplateHandling`

### Test Methods
- Format: `test_<function_name>_<expected_behavior>`
- Examples: `test_apply_ica_masks_basic`, `test_auto_detection_with_chat_template`

## Debugging Tests

### Running Individual Tests

```bash
# Run single test with maximum verbosity
poetry run pytest tests/test_ica_masking.py::TestICAMasking::test_apply_ica_masks_basic -vv

# Run test with Python debugger
poetry run pytest tests/test_ica_masking.py::TestICAMasking::test_apply_ica_masks_basic --pdb

# Run test with output capture disabled
poetry run pytest tests/test_ica_masking.py::TestICAMasking::test_apply_ica_masks_basic -s
```

### Test Logging

Tests use Python's logging module. To see log output:

```bash
# Show log output during tests
poetry run pytest --log-cli-level=INFO

# Show debug logs
poetry run pytest --log-cli-level=DEBUG
```

## Continuous Integration

### Pre-commit Testing

```bash
# Run tests before committing
poetry run pytest

# Run with coverage report
poetry run pytest --cov=src/functionalnetworkssft --cov-report=term-missing
```

### Platform Testing

The `test_installation.py` script verifies:
- Cross-platform compatibility
- Device detection (CPU/CUDA/MPS)
- Dependency installation
- Model loading capabilities

## Test Dependencies

Core testing dependencies (included in main dependencies):
- `pytest>=7.0.0` - Main testing framework
- `unittest` - Python standard library testing
- `torch` - PyTorch for model testing
- `transformers` - Hugging Face transformers
- `numpy` - Numerical operations

## Common Test Patterns

### Mock Model Creation
```python
from tests.test_ica_masking import MockModel
model = MockModel(num_layers=2, hidden_size=768)
```

### Temporary File Testing
```python
import tempfile
with tempfile.TemporaryDirectory() as temp_dir:
    # Test file operations
```

### Device-Agnostic Testing
```python
# Tests automatically detect and use appropriate device
device = torch.device("cpu")  # Fallback for CI environments
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **Device Errors**: Tests fallback to CPU if CUDA/MPS unavailable
3. **Memory Issues**: Reduce batch sizes in test configurations
4. **Path Issues**: Tests automatically handle project root path setup

### Getting Help

- Check test output for specific error messages
- Run individual tests to isolate issues
- Use verbose mode (`-v`) for detailed output
- Check installation with `python tests/test_installation.py`
