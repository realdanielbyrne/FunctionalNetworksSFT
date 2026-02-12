# Continual Learning Framework Assessment

**Date**: 2026-01-23
**Purpose**: Comprehensive evaluation of the FNSFT continual learning framework for comparing ICA-based functional network masking against DOC paper baselines

---

## Executive Summary

The FunctionalNetworksSFT continual learning framework implements the experimental protocol from the DOC paper (Zhang et al., 2025). However, **critical implementation bugs** currently prevent valid comparison experiments. This document details the framework's structure, validates DOC protocol parity, and provides a prioritized improvement plan.

**Status**: NOT READY FOR EXPERIMENTS - Critical fixes required

---

## 1. Documentation Review

### Reference Papers
- **DOC paper**: `docs/Dynamic_Orthogonal_Continual_Fine-Tuning_DOC.md`
  - Defines task orders O1-O6, metrics (AA, BWT, FWT)
  - Baselines: LoRA, EWC, LwF, O-LoRA, DOC
- **ICA functional networks paper**: `docs/Brain-Inspired_Exploration_of_Functional_Networks_and_Key_Neurons_in_LLMs.md`
  - Motivates functional network identification via ICA
  - Key finding: <2% of neurons form critical functional networks

---

## 2. Framework Architecture

### Directory Structure
```
src/functionalnetworkssft/continual_learning/
├── evaluation.py           # Main evaluation pipeline (443 lines)
├── metrics.py              # CL metrics: AA, BWT, FWT (184 lines)
├── utils.py                # Table generation utilities (140 lines)
├── methods/
│   ├── base.py             # Abstract base class (105 lines)
│   ├── lora_baseline.py    # Vanilla LoRA baseline (31 lines)
│   ├── ewc.py              # Elastic Weight Consolidation (132 lines)
│   ├── lwf.py              # Learning without Forgetting (152 lines)
│   ├── o_lora.py           # Orthogonal LoRA (205 lines)
│   ├── doc.py              # Dynamic Orthogonal Continual (342 lines)
│   └── ica_networks.py     # ICA-based Functional Networks (168 lines)
└── datasets/
    ├── config.py           # Dataset and task order configurations (331 lines)
    ├── loaders.py          # Dataset loading utilities (134 lines)
    └── prompts.py          # Task-specific prompt templates (164 lines)

experiments/continual_learning/
├── configs/                # Model configuration YAML files
├── scripts/
│   ├── run_all_baselines.sh
│   ├── run_ica_method.sh
│   └── generate_tables.py
└── results/                # Results storage (.gitkeep)
```

### CLI Entry Points
```bash
poetry run fnsft-cl-eval    # Run evaluation
poetry run fnsft-cl-tables  # Generate comparison tables
```

---

## 3. DOC Protocol Parity Analysis

### 3.1 Task Orders ✅ VERIFIED

All 6 task orders from DOC paper Table 7 are correctly defined in `datasets/config.py:252-307`:

| Order | Type | Tasks | Count | Status |
|-------|------|-------|-------|--------|
| order_1 | Standard | dbpedia → amazon → yahoo → ag_news | 4 | ✅ Exact match |
| order_2 | Standard | dbpedia → amazon → ag_news → yahoo | 4 | ✅ Exact match |
| order_3 | Standard | yahoo → amazon → ag_news → dbpedia | 4 | ✅ Exact match |
| order_4 | Long Chain | mnli → cb → wic → ... → yahoo | 15 | ✅ Exact match |
| order_5 | Long Chain | multirc → boolq → wic → ... → yahoo | 15 | ✅ Exact match |
| order_6 | Long Chain | yelp → amazon → mnli → ... → wic | 15 | ✅ Exact match |

### 3.2 Datasets ✅ VERIFIED

All 15 datasets from DOC paper are configured in `datasets/config.py`:

| Category | Datasets | Status |
|----------|----------|--------|
| CL Benchmark | ag_news, amazon, yelp, dbpedia, yahoo | ✅ Complete |
| GLUE | mnli, qqp, rte, sst2 | ✅ Complete |
| SuperGLUE | wic, cb, copa, boolq, multirc | ✅ Complete |
| Additional | imdb | ✅ Complete |

### 3.3 Metrics ✅ VERIFIED

Formulas in `metrics.py:70-135` match DOC paper exactly:

```
AA(T)  = (1/T) * Σ_{t=1}^{T} a_{t,T}           # Average Accuracy
BWT(T) = (1/(T-1)) * Σ_{t=1}^{T-1} (a_{t,T} - a_{t,t})  # Backward Transfer
FWT(T) = (1/(T-1)) * Σ_{t=2}^{T} (a_{t,t} - baseline_t)  # Forward Transfer
```

### 3.4 Baseline Methods ✅ VERIFIED

All DOC paper baselines implemented in `methods/`:

| Method | File | Implementation |
|--------|------|----------------|
| LoRA Baseline | `lora_baseline.py` | Standard fine-tuning, no CL mechanism |
| EWC | `ewc.py` | Fisher information regularization |
| LwF | `lwf.py` | Knowledge distillation |
| O-LoRA | `o_lora.py` | SVD-based subspace orthogonality |
| DOC | `doc.py` | Gradient direction tracking + projection |
| ICA Networks | `ica_networks.py` | FNSFT functional network masking |

### 3.5 Table Generation ✅ VERIFIED

`generate_tables.py:27-98` produces DOC paper format:

**Accuracy Table (Tables 1 & 2):**
```
| Method       | O1   | O2   | O3   | Std Avg | O4   | O5   | O6   | Long Avg |
```

**BWT/FWT Table (Table 3):**
```
| Method       | Std BWT | Std FWT | Long BWT | Long FWT |
```

---

## 4. Critical Implementation Bugs

### 🔴 Bug 1: Evaluation Data Leakage (CRITICAL)

**Location**: `datasets/loaders.py:72-79`

**Problem**: Test data includes ground-truth answers in input, making evaluation invalid.

```python
# loaders.py:72-79
def format_fn(x):
    return format_example(x, config, include_answer=True)  # BUG: Always True

train_data = train_data.map(format_fn, ...)
test_data = test_data.map(format_fn, ...)  # Test data ALSO gets answers!
```

**Impact**: Models see answers during evaluation. All accuracy metrics are artificially inflated and meaningless.

**Fix Required**:
```python
def format_train_fn(x):
    return format_example(x, config, include_answer=True)

def format_test_fn(x):
    return format_example(x, config, include_answer=False)

train_data = train_data.map(format_train_fn, ...)
test_data = test_data.map(format_test_fn, ...)
```

---

### 🔴 Bug 2: Ground-Truth Extraction Unreliable (CRITICAL)

**Location**: `evaluation.py:151-161`

**Problem**: Ground-truth is extracted by scanning decoded tokens for ANY matching label string.

```python
# evaluation.py:154-161
labels = example["labels"]
ground_truth = None
for ans in valid_answers:
    if ans.lower() in tokenizer.decode(labels).lower():  # BUG: Unreliable
        ground_truth = ans.lower()
        break
```

**Impact**: Since prompts include "Options: entailment, neutral, contradiction", the first option often matches incorrectly.

**Fix Required**: Store ground-truth label separately during dataset preparation, not extracted from tokenized text.

---

### 🔴 Bug 3: DOC Gradient Projection Never Called (CRITICAL)

**Location**: `evaluation.py:200-206`

**Problem**: Training loop does not call `apply_gradient_projection()`.

```python
# evaluation.py:200-206
loss = cl_method.compute_loss(batch, cl_method.current_task_idx)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# MISSING: cl_method.apply_gradient_projection()  <-- BUG
optimizer.step()
```

**Impact**: DOC degenerates to vanilla LoRA. All DOC results are invalid.

**Fix Required**:
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
if hasattr(cl_method, 'apply_gradient_projection'):
    cl_method.apply_gradient_projection()  # Add this line
optimizer.step()
```

---

### 🟡 Bug 4: FWT Baselines Never Computed (HIGH)

**Location**: `evaluation.py:221-222`

**Problem**: `compute_baselines` parameter exists but is unused.

```python
def run_continual_learning_evaluation(
    ...
    compute_baselines: bool = False,  # Never used
    ...
):
```

**Impact**: FWT is always 0.0 because `baseline_accuracies` remains zeros.

**Fix Required**: Implement single-task fine-tuning baseline runs before CL experiments.

---

### 🟡 Bug 5: ICA Method No-Op Without Templates (HIGH)

**Location**: `methods/ica_networks.py:117-124`

**Problem**: Masking only applied if `ica_template_path` provided.

```python
# ica_networks.py:117
if task_idx > 0 and self.ica_mask is not None:  # Requires template
    protected = self._select_components_to_protect(task_idx, task_data)
    self.ica_mask.apply_component_masks(...)
```

**Impact**: Default ICA runs apply NO masking, equivalent to LoRA baseline.

**Fix Required**: Either require templates or add warning when no masks applied.

---

### 🟡 Bug 6: T5 Model Mode Mismatch (MEDIUM)

**Location**: `evaluation.py:102-116`

**Problem**: All models loaded as `AutoModelForCausalLM` with `TaskType.CAUSAL_LM`.

**Impact**: T5 is a seq2seq model requiring `AutoModelForSeq2SeqLM` and `TaskType.SEQ_2_SEQ_LM`.

**Fix Required**: Detect T5 models and load with correct class and task type.

---

## 5. Additional Deviations (Non-Blocking)

### DOC/O-LoRA Algorithm Fidelity

The implementations use SVD over parameter/gradient snapshots rather than DOC paper's Online PCA with amnesic factor and tracking. This is a **simplification** that may not capture drifting functional directions as accurately.

**Specific Differences**:
- DOC paper uses CCIPCA (Candid Covariance-free Incremental PCA)
- Implementation uses batch SVD after task completion
- DOC paper uses amnesic factor l=2 and tracking factor ε
- Implementation uses fixed subspace_fraction

---

## 6. Improvement Plan (Prioritized)

### Phase 1: Critical Bug Fixes (Required Before Any Experiments)

| Priority | Issue | File | Lines | Fix |
|----------|-------|------|-------|-----|
| P0 | Evaluation leakage | `loaders.py` | 72-79 | Separate train/test formatting |
| P0 | Ground-truth extraction | `evaluation.py` | 151-161 | Store labels during data prep |
| P0 | DOC projection missing | `evaluation.py` | 200-206 | Add `apply_gradient_projection()` call |

### Phase 2: Missing Functionality (Required for Valid Metrics)

| Priority | Issue | File | Fix |
|----------|-------|------|-----|
| P1 | FWT baselines | `evaluation.py` | Implement single-task baseline runs |
| P1 | ICA templates | `ica_networks.py` | Add warnings, require templates for CL |

### Phase 3: Model Support (Required for Full DOC Parity)

| Priority | Issue | File | Fix |
|----------|-------|------|-----|
| P2 | T5 model mode | `evaluation.py` | Detect and load as seq2seq |

### Phase 4: Algorithm Fidelity (Optional Enhancement)

| Priority | Issue | File | Fix |
|----------|-------|------|-----|
| P3 | Online PCA | `doc.py` | Replace batch SVD with CCIPCA |
| P3 | Drift tracking | `doc.py` | Add amnesic factor |

---

## 7. Validation Checklist (Post-Fix)

After implementing fixes, verify:

1. **Random Model Sanity Check**
   - Load untrained model, evaluate on test set
   - Expected: Near-chance accuracy (e.g., ~25% for 4-class, ~50% for binary)
   - If accuracy >> chance: evaluation still has leakage

2. **DOC Projection Verification**
   - Add logging in `apply_gradient_projection()`
   - Verify projection applied for task_idx > 0
   - Compare gradients before/after projection

3. **FWT Baseline Validation**
   - Run single-task fine-tuning for each task
   - Store baseline accuracies
   - Verify FWT is non-zero after CL runs

4. **ICA Masking Verification**
   - Enable verbose logging in `ica_networks.py`
   - Verify "Protecting components [...]" appears for task_idx > 0
   - Check hook count in `ica_mask.mask_handles`

---

## 8. Conclusion

### Current State
- **Protocol Parity**: ✅ Task orders, datasets, metrics, table formats match DOC paper
- **Implementation**: ❌ Critical bugs prevent valid experiments

### Path Forward
1. Fix critical bugs in Phase 1 (evaluation leakage, ground-truth, DOC projection)
2. Implement missing functionality in Phase 2 (FWT baselines, ICA templates)
3. Run validation checklist to confirm fixes
4. Then proceed with ICA Networks vs DOC comparison experiments

**Do not run experiments or trust results until all Phase 1 items are complete.**

---

## Appendix: Key File References

| Purpose | File | Lines |
|---------|------|-------|
| Task Orders | `datasets/config.py` | 252-307 |
| Metrics Formulas | `metrics.py` | 70-135 |
| Evaluation Loop | `evaluation.py` | 168-212 |
| Data Loading BUG | `loaders.py` | 72-79 |
| Ground-Truth BUG | `evaluation.py` | 151-161 |
| DOC Projection BUG | `evaluation.py` | 200-206 |
| DOC Method | `methods/doc.py` | 262-283 |
| ICA Method | `methods/ica_networks.py` | 111-124 |
| Table Generation | `scripts/generate_tables.py` | 27-98 |
