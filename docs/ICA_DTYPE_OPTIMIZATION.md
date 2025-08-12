# ICA Dtype Optimization

## Overview

The ICA mask computation in FunctionalNetworksSFT now supports configurable data types to optimize performance and memory usage. Previously, all ICA computations were performed in float32 for maximum numerical stability. Now you can choose different precisions based on your performance requirements and model characteristics.

## Background

**Question**: Does the ICA mask computation need to be performed in float32 when the model isn't float32? Can we reduce the precision to improve performance?

**Answer**: Yes, we can reduce precision to improve performance, but with trade-offs in numerical stability.

## New Feature: `ica_dtype` Parameter

### Configuration Options

- **`None` or `"float32"` (default)**: Maximum numerical stability, uses float32 for all ICA computations
- **`"auto"`**: Automatically matches model dtype but uses float32 for half-precision models to maintain stability
- **`"float16"`**: Reduced precision, ~50% memory savings, potential numerical issues
- **`"bfloat16"`**: Better numerical stability than float16, ~50% memory savings

### Performance Benefits

| Dtype | Memory Usage | Numerical Stability | Recommended Use |
|-------|-------------|-------------------|-----------------|
| float32 | 100% (baseline) | Highest | Production, critical applications |
| auto | Varies by model | High | General use, automatic optimization |
| bfloat16 | ~50% | Medium-High | Memory-constrained environments |
| float16 | ~50% | Medium | Experimental, with careful validation |

## Usage Examples

### Command Line

```bash
# Use automatic dtype selection
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --mask_mode key \
    --ica_dtype auto

# Use reduced precision for memory savings
python -m functionalnetworkssft.fnsft_trainer \
    --model_name_or_path microsoft/DialoGPT-medium \
    --mask_mode key \
    --ica_dtype bfloat16
```

### Configuration File

```yaml
# config.yaml
model_name_or_path: "microsoft/DialoGPT-medium"
mask_mode: "key"
ica_dtype: "auto"  # or "float32", "float16", "bfloat16"
```

### Programmatic Usage

```python
from functionalnetworkssft.ica_mask import ICAMask

# Default (float32)
ica_mask = ICAMask()

# Automatic dtype selection
ica_mask = ICAMask(ica_dtype="auto")

# Explicit reduced precision
ica_mask = ICAMask(ica_dtype="bfloat16")
```

## Implementation Details

### Dtype Selection Logic

The `_get_ica_dtype()` method determines the optimal dtype:

1. **`None`/`"float32"`**: Always use float32
2. **`"auto"`**: Use float32 for half-precision models, match dtype for full-precision models
3. **`"float16"`/`"bfloat16"`**: Use specified reduced precision
4. **Invalid values**: Fall back to float32 with warning

### Memory Optimization

- **float16/bfloat16**: ~50% memory reduction during ICA computation
- **Activation capture**: Converts activations to target dtype immediately after capture
- **NumPy processing**: All subsequent ICA operations use the selected precision

### Numerical Considerations

- **ICA algorithm**: FastICA can be sensitive to precision, especially with extreme values
- **Standardization**: Mean/std calculations may lose precision with float16
- **Matrix decomposition**: SVD operations in ICA may be affected by reduced precision

## Recommendations

### Production Use
- **Default**: Use `ica_dtype=None` (float32) for maximum stability
- **Memory-constrained**: Use `ica_dtype="auto"` for automatic optimization
- **Large models**: Consider `ica_dtype="bfloat16"` for memory savings

### Experimental Use
- **Research**: Test `ica_dtype="float16"` with careful validation
- **Benchmarking**: Use the provided demo script to evaluate performance

### Model-Specific Guidelines

| Model Type | Recommended ica_dtype | Rationale |
|------------|----------------------|-----------|
| float32 models | `"auto"` or `None` | Match precision or use stable default |
| float16 models | `"auto"` | Automatically uses float32 for stability |
| bfloat16 models | `"auto"` or `"bfloat16"` | Match precision or use stable default |
| Quantized models | `"auto"` | Let system choose optimal precision |

## Testing and Validation

### Performance Demo
Run the included demonstration script:
```bash
python examples/ica_dtype_performance_demo.py
```

### Unit Tests
```bash
python -m pytest tests/test_ica_dtype_optimization.py -v
```

### Validation Checklist
- [ ] ICA masks are generated successfully
- [ ] Memory usage is reduced as expected
- [ ] Training converges normally
- [ ] Model performance is maintained
- [ ] No numerical warnings or errors

## Troubleshooting

### Common Issues

**Numerical instability with float16**:
- Switch to `bfloat16` or `auto`
- Check for extreme activation values
- Consider enabling `clip_activations=True`

**Memory not reduced as expected**:
- Verify the dtype is being applied correctly
- Check that model activations are large enough to see difference
- Monitor peak memory usage during ICA computation

**ICA computation fails**:
- Fall back to default `float32`
- Check for NaN/Inf values in activations
- Reduce number of ICA components if needed

## Future Enhancements

- **Adaptive precision**: Automatically detect numerical issues and fall back to higher precision
- **Mixed precision ICA**: Use different precisions for different stages of computation
- **Platform optimization**: Optimize dtype selection based on hardware capabilities
