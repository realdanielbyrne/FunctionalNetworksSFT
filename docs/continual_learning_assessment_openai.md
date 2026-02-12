# Continual Learning Framework Assessment (FNSFT vs DOC)

## Scope
This assessment covers the continual learning experimental framework in
`src/functionalnetworkssft/continual_learning/` and
`experiments/continual_learning/`, and checks protocol parity against the DOC
paper (Zhang et al., 2025) and the ICA functional networks motivation.

## Doc Review Notes
- DOC paper: `docs/Dynamic_Orthogonal_Continual_Fine-Tuning_DOC.md`
- ICA functional networks paper: `docs/Brain-Inspired_Exploration_of_Functional_Networks_and_Key_Neurons_in_LLMs.md`

The DOC paper defines task orders O1-O6, metrics (AA, BWT, FWT), and compares
LoRA, EWC, LwF, O-LoRA, DOC, plus additional baselines. The ICA paper motivates
functional network identification and masking as an intervention, which informs
FNSFT's ICA masking approach.

## Parity Check Against DOC Protocol
### Task Orders
- Implemented orders 1-6 in
  `src/functionalnetworkssft/continual_learning/datasets/config.py`.
- Orders 1-3 map to the CL Benchmark (AG News, Yelp, Amazon, DBPedia, Yahoo).
- Orders 4-6 map to the long chain tasks (GLUE + SuperGLUE + IMDB + CL
  Benchmark), matching DOC Table 7.

### Baseline Methods
- Implemented baselines in `src/functionalnetworkssft/continual_learning/methods/`:
  - `lora_baseline.py`, `ewc.py`, `lwf.py`, `o_lora.py`, `doc.py`,
    `ica_networks.py`.
- CLI scripts include all requested methods:
  `experiments/continual_learning/scripts/run_all_baselines.sh`.

### Metrics
- AA, BWT, FWT definitions match DOC equations:
  `src/functionalnetworkssft/continual_learning/metrics.py`.
- FWT requires per-task baseline accuracies, which are not computed in the
  current evaluation loop.

### Table Generation
- `experiments/continual_learning/scripts/generate_tables.py` matches DOC
  formats:
  - Table 1/2: Method | O1 | O2 | O3 | Std Avg | O4 | O5 | O6 | Long Avg
  - Table 3: Method | Std BWT | Std FWT | Long BWT | Long FWT
- Includes methods: LoRA, EWC, LwF, O-LoRA, DOC, ICA_Networks.

## Critical Gaps / Issues (Blocking Fair Comparison)
1. **Evaluation leakage in test data**
   - Test data is formatted with answers appended, so evaluation inputs include
     the ground-truth answer. This invalidates accuracy and all derived metrics.
   - Evidence:
     - `src/functionalnetworkssft/continual_learning/datasets/loaders.py`
     - `src/functionalnetworkssft/continual_learning/evaluation.py`

2. **Ground-truth extraction is unreliable**
   - The evaluation logic scans decoded labels for any valid answer string.
     Since the prompt includes the options list, this can match the wrong
     label, frequently selecting the first option.
   - Evidence: `src/functionalnetworkssft/continual_learning/evaluation.py`

3. **DOC gradient projection never applied**
   - DOC implements gradient projection but the training loop never calls
     `apply_gradient_projection()`, so DOC degenerates to vanilla LoRA.
   - Evidence:
     - `src/functionalnetworkssft/continual_learning/methods/doc.py`
     - `src/functionalnetworkssft/continual_learning/evaluation.py`

4. **FWT baselines never computed**
   - `compute_baselines` is unused; baseline accuracies remain zero. FWT
     values are meaningless unless baselines are computed.
   - Evidence:
     - `src/functionalnetworkssft/continual_learning/evaluation.py`
     - `src/functionalnetworkssft/continual_learning/metrics.py`
     - `experiments/continual_learning/scripts/run_all_baselines.sh`

5. **ICA method is a no-op without templates**
   - `ica_networks` only applies masking if `ica_template_path` is provided.
     Default runs do not apply any mask.
   - Evidence: `src/functionalnetworkssft/continual_learning/methods/ica_networks.py`

6. **Model mismatch for T5**
   - The evaluation uses `AutoModelForCausalLM` and LoRA `TaskType.CAUSAL_LM`.
     DOC uses T5-Large in seq2seq mode; this mismatch likely breaks parity.
   - Evidence: `src/functionalnetworkssft/continual_learning/evaluation.py`

## Additional Deviations (Non-Blocking but Important)
- O-LoRA and DOC are implemented using SVD over parameter/gradient snapshots,
  not the DOC paper's online PCA for drifted functional directions. This is
  likely not a faithful reproduction of DOC's mechanics.

## Plan to Improve (No Code Changes Yet)
1. **Fix evaluation correctness**
   - Ensure test inputs exclude answers, and ground-truth labels are read from
     the raw dataset rather than decoded strings.
   - Confirm accuracy computation against simple sanity checks (random model
     vs. expected near-chance performance).

2. **Implement and validate FWT baselines**
   - Add baseline runs for each task (single-task fine-tuning) and store
     per-task accuracies in `baseline_accuracies`.
   - Recompute FWT for all methods.

3. **Activate DOC gradient projection**
   - Call `apply_gradient_projection()` in the training loop after
     `loss.backward()` and before `optimizer.step()`.
   - Add a smoke test to verify that projection is executed for task > 0.

4. **Clarify ICA masking usage**
   - Require or auto-provide ICA templates for `ica_networks` runs.
   - Add explicit logging to warn when no masks are applied.

5. **Resolve model-mode parity for T5**
   - Either support seq2seq models (T5) with `AutoModelForSeq2SeqLM` and
     matching LoRA task type, or remove T5 from the evaluation matrix.

6. **Re-evaluate DOC/O-LoRA fidelity**
   - Compare current implementations to DOC/O-LoRA algorithms and decide if
     additional alignment (online PCA, drift tracking) is needed for fair
     comparison.

## Outcome Summary
- The framework has the right task orders, metrics definitions, and table
  formats to mirror DOC.
- Several critical issues in evaluation and DOC implementation currently
  prevent a fair, DOC-style comparison between FNSFT (ICA masking) and the
  baselines.
- Addressing the items in the plan above is required before trusting AA/BWT/FWT
  tables or claiming parity with DOC results.
