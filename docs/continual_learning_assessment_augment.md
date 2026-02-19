# Continual Learning Experimental Framework Evaluation

## Overview

This document evaluates the FNSFT continual learning framework against the DOC paper (Zhang et al., 2025) experimental protocol. The framework aims to compare ICA-based functional network masking against established continual learning baselines.

## Reference Papers

- **DOC**: "Dynamic Orthogonal Continual Fine-Tuning for Mitigating Catastrophic Forgetting" (Zhang et al., 2025)
- **Functional Networks**: "Brain-Inspired Exploration of Functional Networks and Key Neurons in LLMs" (Liu et al., 2025)

---

## Current Implementation Status

### ✅ Correctly Implemented Components

| Component | Location | Status |
|-----------|----------|--------|
| Average Accuracy (AA) | `metrics.py` | Matches DOC Eq. 15 |
| Backward Transfer (BWT) | `metrics.py` | Matches DOC Eq. 15 |
| Forward Transfer (FWT) | `metrics.py` | Matches DOC Eq. 15 |
| LoRA Baseline | `methods/lora_baseline.py` | Complete |
| EWC Method | `methods/ewc.py` | Complete |
| LwF Method | `methods/lwf.py` | Complete |
| O-LoRA Method | `methods/o_lora.py` | Complete |
| DOC Method | `methods/doc.py` | Partial (see issues) |
| ICA Networks | `methods/ica_networks.py` | Complete |
| Table Generation | `utils.py` | Matches DOC Tables 1-3 format |
| Long Chain Orders | `datasets/config.py` | 15 tasks matching DOC |
| CLI Commands | `pyproject.toml` | `fnsft-cl-eval`, `fnsft-cl-tables` |
| Hyperparameters | `evaluation.py` | Matches DOC paper |

### ❌ Critical Issues

#### 1. Standard CL Benchmark Missing Yelp Dataset

**Problem**: DOC paper uses 5 datasets for standard benchmark, implementation has 4.

**Current** (`datasets/config.py`):
```python
TASK_ORDERS = {
    "order_1": ["dbpedia", "amazon", "yahoo", "ag_news"],  # 4 tasks
    "order_2": ["dbpedia", "amazon", "ag_news", "yahoo"],
    "order_3": ["yahoo", "amazon", "ag_news", "dbpedia"],
}
```

**DOC Paper** (Table 7, Appendix B):
- Order 1: AG News → Yelp → Amazon → DBPedia → Yahoo (5 tasks)
- Order 2: DBPedia → Yahoo → AG News → Amazon → Yelp
- Order 3: Yelp → Yahoo → Amazon → DBPedia → AG News

**Impact**: Results will not be directly comparable to DOC paper.

#### 2. DOC Gradient Projection Not Integrated

**Problem**: The DOC method implements `apply_gradient_projection()` but it's never called.

**Current** (`evaluation.py` lines 201-206):
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()  # Gradient projection should be called before this
```

**Required**: Call `cl_method.apply_gradient_projection()` after `backward()` and before `step()`.

**Impact**: DOC method runs as standard LoRA without its key innovation.

### ⚠️ Minor Issues

1. **FWT Baselines Not Computed**: `--no_baselines` is default; baseline accuracy needed for FWT
2. **Training Steps Variance**: `llama-3.2-1b` uses 500 steps vs 1000 in DOC paper
3. **LoRA Target Modules**: Could include MLP layers (`down_proj`, `up_proj`)

---

## Improvement Plan

### Phase 1: Critical Fixes

#### Task 1.1: Fix Standard CL Benchmark Task Orders
- **File**: `src/functionalnetworkssft/continual_learning/datasets/config.py`
- **Action**: Update `order_1`, `order_2`, `order_3` to include 5 tasks with Yelp
- **Reference**: DOC paper Table 7 (Appendix B)

#### Task 1.2: Integrate DOC Gradient Projection
- **File**: `src/functionalnetworkssft/continual_learning/evaluation.py`
- **Action**: Add conditional call to `apply_gradient_projection()` in `train_on_task()`
- **Implementation**:
  ```python
  loss.backward()
  if hasattr(cl_method, 'apply_gradient_projection'):
      cl_method.apply_gradient_projection()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  optimizer.step()
  ```

### Phase 2: Baseline Computation

#### Task 2.1: Implement FWT Baseline Computation
- **File**: `src/functionalnetworkssft/continual_learning/evaluation.py`
- **Action**: Add function to compute per-task baseline accuracy via standard fine-tuning
- **Logic**: For each task, train fresh LoRA and record accuracy as baseline

### Phase 3: Validation

#### Task 3.1: Run Comparison Experiments
- Execute all methods on all task orders
- Generate comparison tables
- Verify metrics are in expected ranges per DOC paper

#### Task 3.2: Document Results
- Add results to experiments folder
- Create visualization of ICA Networks vs baselines

---

## Expected Table Output Format

### Table 1/2: Average Accuracy (AA)

| Method | O1 | O2 | O3 | Std Avg | O4 | O5 | O6 | Long Avg |
|--------|----|----|----|---------|----|----|----|----------|
| LORA | - | - | - | - | - | - | - | - |
| EWC | - | - | - | - | - | - | - | - |
| LWF | - | - | - | - | - | - | - | - |
| O_LORA | - | - | - | - | - | - | - | - |
| DOC | - | - | - | - | - | - | - | - |
| ICA_NETWORKS | - | - | - | - | - | - | - | - |

### Table 3: BWT/FWT

| Method | Std BWT | Std FWT | Long BWT | Long FWT |
|--------|---------|---------|----------|----------|
| LORA | - | - | - | - |
| ... | ... | ... | ... | ... |

---

## Commands

```bash
# Run all baselines on all orders
./experiments/continual_learning/scripts/run_all_baselines.sh llama-7b ./results

# Run single method
poetry run fnsft-cl-eval --model llama-7b --method ica_networks --task_order order_1

# Generate comparison tables
poetry run fnsft-cl-tables --results_dir ./results --model llama-7b
```

---

## Conclusion

The framework is approximately **85% complete**. After implementing the two critical fixes (task orders and DOC gradient projection), the framework will be capable of generating DOC-style comparison tables that fairly evaluate ICA-based functional network masking against EWC, O-LoRA, and DOC baselines.

