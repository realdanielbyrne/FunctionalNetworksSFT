# Functional Network Masking for Continual Learning in Large Language Models: An ICA-Based Approach

**Daniel Byrne**

---

## Abstract

Catastrophic forgetting remains a fundamental challenge in the sequential fine-tuning of Large Language Models (LLMs), where learning new tasks degrades performance on previously acquired knowledge. We propose a novel approach to continual learning that draws inspiration from cognitive neuroscience, leveraging Independent Component Analysis (ICA) to identify functional networks within LLM architectures—groups of neurons that exhibit coordinated activation patterns analogous to functional brain networks observed in fMRI studies. By selectively masking these identified functional networks during sequential fine-tuning, we aim to preserve critical pre-trained representations while enabling adaptation to new tasks. Our framework implements two complementary masking strategies: *lesion mode*, which ablates specific functional networks to prevent their modification, and *preserve mode*, which restricts training to selected functional networks to isolate task-specific learning. We evaluate our approach against established continual learning baselines—including Elastic Weight Consolidation (EWC), Learning without Forgetting (LwF), Orthogonal LoRA (O-LoRA), and Dynamic Orthogonal Continual fine-tuning (DOC)—following the experimental protocol of Zhang et al. (2025) across six task orders spanning 15 text classification datasets. We hypothesize that ICA-based functional network masking can achieve competitive or superior performance in mitigating catastrophic forgetting while offering interpretable, neuroscience-grounded insights into the organization of knowledge within LLMs. Experimental results are forthcoming and will be reported upon completion of the evaluation protocol.

---

## 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks, from text classification and question answering to complex reasoning and code generation (Brown et al., 2020; Touvron et al., 2023). However, deploying these models in real-world scenarios often requires sequential adaptation to new tasks or domains—a setting where *catastrophic forgetting* poses a significant challenge (McCloskey & Cohen, 1989; French, 1999). When fine-tuned on a new task, neural networks tend to overwrite the representations learned from previous tasks, leading to severe performance degradation on earlier objectives.

The continual learning community has developed numerous strategies to address this challenge, broadly categorized into regularization-based methods that constrain parameter updates (Kirkpatrick et al., 2017; Li & Hoiem, 2017), architecture-based methods that allocate dedicated capacity for each task (Rusu et al., 2016; Mallya & Lazebnik, 2018), and replay-based methods that maintain access to historical data (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2019). More recently, parameter-efficient fine-tuning (PEFT) approaches, particularly Low-Rank Adaptation (LoRA) (Hu et al., 2022), have opened new avenues for continual learning in LLMs by constraining updates to low-rank subspaces (Wang et al., 2023; Zhang et al., 2025).

Despite these advances, existing methods often lack a principled understanding of *which* parameters or neurons are critical for preserving specific capabilities. Regularization-based approaches like EWC estimate parameter importance through the Fisher information matrix, but this provides only a local approximation that degrades over multiple tasks. Orthogonal projection methods like O-LoRA and DOC operate in the parameter space without direct insight into the functional organization of the network's representations.

A compelling parallel exists in cognitive neuroscience, where decades of research have revealed that the human brain organizes its processing into *functional networks*—spatially distributed but temporally coordinated groups of neurons that collectively support specific cognitive functions (Smith et al., 2009; Yeo et al., 2011). These functional networks, identified through Independent Component Analysis (ICA) of functional magnetic resonance imaging (fMRI) data, exhibit remarkable consistency across individuals and tasks, suggesting a fundamental organizational principle of neural information processing.

Recent work by Liu et al. (2025) has demonstrated that analogous functional networks exist within LLMs. By applying ICA to the activation patterns of MLP neurons across transformer layers, they identified groups of neurons that exhibit coordinated activation patterns reminiscent of brain functional networks. Critically, they showed that masking fewer than 2% of neurons belonging to key functional networks severely degrades model performance, while preserving approximately 10% of neurons within these networks maintains near-baseline capability. These findings suggest that LLMs, like biological brains, organize their computations through sparse, functionally specialized networks.

In this work, we extend these neuroscience-inspired findings to the domain of continual learning. We propose *Functional Network Masking* (FNM), a method that leverages ICA-identified functional networks to selectively control which portions of an LLM are modified during sequential fine-tuning. Our approach operates on the hypothesis that by identifying and protecting the functional networks most critical to previously learned tasks, we can mitigate catastrophic forgetting while maintaining the capacity to learn new tasks through the remaining, less critical network components.

We evaluate our approach using the comprehensive experimental protocol established by Zhang et al. (2025) in their Dynamic Orthogonal Continual (DOC) fine-tuning framework, which provides a rigorous benchmark across six task orders spanning standard and long-chain continual learning scenarios. This enables direct comparison against state-of-the-art baselines including EWC, LwF, O-LoRA, and DOC itself.

**Our contributions are as follows:**

1. We propose a novel continual learning method for LLMs that leverages ICA-based functional network identification to selectively mask neurons during sequential fine-tuning, providing a neuroscience-grounded approach to mitigating catastrophic forgetting.
2. We implement two complementary masking strategies—lesion and preserve modes—that offer different trade-offs between knowledge preservation and new task learning capacity.
3. We develop a template-based system for pre-computing and reusing functional network masks, enabling efficient deployment across multiple continual learning scenarios.
4. We provide a comprehensive evaluation framework that directly compares our approach against established baselines following the DOC experimental protocol.

The remainder of this paper is organized as follows. Section 2 reviews related work in continual learning, functional networks, and ICA applications. Section 3 details our specific contributions. Section 4 presents our methodology. Section 5 describes the experimental setup. Sections 6–8 present results, discussion, and conclusions, respectively.

---

## 2. Related Work

### 2.1 Continual Learning in Neural Networks

Continual learning—also known as lifelong learning or sequential learning—addresses the challenge of training models on a sequence of tasks without forgetting previously learned knowledge (Parisi et al., 2019; De Lange et al., 2021). The field has developed three primary families of approaches.

**Regularization-based methods** add penalty terms to the loss function that discourage changes to parameters deemed important for previous tasks. Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017) estimates parameter importance using the diagonal of the Fisher information matrix, penalizing deviations from previously learned parameter values proportional to their estimated importance:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where $F_i$ represents the Fisher information for parameter $\theta_i$ and $\theta_i^*$ denotes the optimal parameters after the previous task. Learning without Forgetting (LwF) (Li & Hoiem, 2017) takes a knowledge distillation approach, using the outputs of the model on new task data (computed before training) as soft targets:

$$\mathcal{L}_{\text{LwF}} = \mathcal{L}_{\text{task}} + \alpha \cdot \text{KL}\left(\text{softmax}(z_{\text{old}}/T) \| \text{softmax}(z_{\text{new}}/T)\right)$$

where $T$ is a temperature parameter that softens the probability distributions. Synaptic Intelligence (SI) (Zenke et al., 2017) tracks the contribution of each parameter to the loss reduction during training, providing an online estimate of parameter importance.

**Architecture-based methods** allocate dedicated model capacity for each task. Progressive Neural Networks (Rusu et al., 2016) add new columns of parameters for each task while freezing previous columns. PackNet (Mallya & Lazebnik, 2018) uses iterative pruning to identify and freeze task-specific subnetworks within a shared architecture. These methods avoid forgetting by construction but suffer from growing computational and memory requirements.

**Replay-based methods** maintain access to data from previous tasks, either through explicit storage or generative models. Experience Replay (Ratcliff, 1990) interleaves stored examples from previous tasks during training. Gradient Episodic Memory (GEM) (Lopez-Paz & Ranzato, 2017) constrains gradient updates to avoid increasing the loss on stored examples. Averaged GEM (A-GEM) (Chaudhry et al., 2019) provides a more efficient approximation by projecting gradients onto the average gradient direction of previous tasks.


### 2.2 Continual Learning for Large Language Models

The emergence of LLMs has introduced unique challenges and opportunities for continual learning. The sheer scale of these models makes full fine-tuning impractical for sequential task adaptation, motivating parameter-efficient approaches.

**LoRA and Parameter-Efficient Fine-Tuning.** Low-Rank Adaptation (LoRA) (Hu et al., 2022) constrains weight updates to low-rank matrices $\Delta W = BA$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ with rank $r \ll d$. This dramatically reduces the number of trainable parameters while maintaining competitive performance. LoRA has become the de facto standard for LLM fine-tuning and provides a natural foundation for continual learning approaches.

**Orthogonal LoRA (O-LoRA).** Wang et al. (2023) proposed O-LoRA, which maintains orthogonal subspaces for different tasks by constraining the LoRA B matrices to be orthogonal to those learned for previous tasks. After each task, the principal directions of the LoRA parameters are computed via SVD, and subsequent updates are projected to be orthogonal to these directions:

$$\mathcal{L}_{\text{O-LoRA}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{ortho}}$$

This approach preserves previous task knowledge by ensuring new learning occurs in orthogonal subspaces, but uses fixed functional directions that may become invalid as the model evolves.

**Dynamic Orthogonal Continual Fine-Tuning (DOC).** Zhang et al. (2025) identified a critical limitation of existing regularization-based methods: *functional direction drift*. As models are fine-tuned sequentially, the parameter space shifts, breaking the locality of linearity in deep neural networks and rendering previously recorded functional directions invalid. DOC addresses this through:

1. **Online PCA-based tracking**: Using modified Candid Covariance-free Incremental PCA (CCIPCA) to continuously track and update functional direction bases as the model evolves.
2. **Orthogonal gradient projection**: Adjusting gradients to be orthogonal to tracked historical functional directions, ensuring new learning does not interfere with previously acquired knowledge.

DOC achieves state-of-the-art results on both standard and long-chain continual learning benchmarks, demonstrating 77.7% average accuracy on LLaMA-7B compared to O-LoRA's 76.5% on the standard benchmark, with significantly reduced backward transfer (BWT of -0.6 vs. -1.9).

### 2.3 Functional Networks in Neural Systems

The concept of functional networks originates in cognitive neuroscience, where functional magnetic resonance imaging (fMRI) studies have revealed that the brain organizes its processing into spatially distributed but temporally coordinated networks (Biswal et al., 1995; Smith et al., 2009). These *functional brain networks* (FBNs) include well-characterized systems such as the default mode network, the executive control network, and the salience network (Yeo et al., 2011; Power et al., 2011).

Independent Component Analysis has been the primary tool for identifying these networks from fMRI data (Beckmann & Smith, 2004; Calhoun et al., 2001). ICA decomposes the observed signals into maximally statistically independent components, each representing a distinct functional network. The spatial maps of these components reveal which brain regions participate in each network, while the temporal profiles indicate when each network is active.

**Functional Networks in LLMs.** Liu et al. (2025) pioneered the application of neuroscience-inspired functional network analysis to LLMs in their work "Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models." They drew a direct analogy between fMRI signals from brain regions and activation patterns of MLP neurons in transformer layers:

- **fMRI signals ↔ MLP neuron outputs**: Both capture neural activity in response to stimuli and can be decomposed using ICA.
- **Brain functional networks ↔ LLM functional networks**: Both represent collections of processing units that exhibit coordinated activation across different inputs.
- **Sparse activation**: Both biological brains (5–10% active neurons) and LLMs exhibit sparse functional network activation.

Their key experimental findings include:

1. **Spatial consistency**: Functional networks identified via group-wise ICA show significant consistency when validated against individual sample analyses, measured by Intersection over Union (IoU) metrics.
2. **Critical neurons**: Masking fewer than 2% of neurons belonging to key functional networks causes severe performance degradation (e.g., accuracy dropping from 0.86 to 0.56 on AG News with GPT-2).
3. **Sufficient networks**: Preserving approximately 10% of neurons within key functional networks maintains near-baseline performance.
4. **Random masking resilience**: Random masking of 15% of neurons has minimal impact, confirming that the identified networks capture genuinely important functional structure.

These findings provide the theoretical foundation for our approach: if functional networks represent the critical organizational structure of LLM computations, then selectively protecting these networks during continual learning should mitigate catastrophic forgetting.

### 2.4 ICA Applications in Deep Learning

Independent Component Analysis has found diverse applications in deep learning beyond the neuroscience-inspired analysis described above. ICA has been used for feature extraction in convolutional neural networks (Le et al., 2011), for understanding learned representations in autoencoders (Hyvärinen & Pajunen, 1999), and for disentangling latent factors in generative models (Khemakhem et al., 2020).

The FastICA algorithm (Hyvärinen & Oja, 2000) is the most widely used variant, employing a fixed-point iteration scheme to maximize non-Gaussianity as a proxy for statistical independence. The algorithm operates in two phases: (1) pre-whitening to remove linear correlations and standardize variance, and (2) iterative optimization of weight vectors to maximize the objective function:

$$J(\mathbf{w}) = \left[\mathbb{E}\{G(\mathbf{w}^T \mathbf{z})\}\right]^2 - \frac{1}{2}\mathbb{E}\{(\mathbf{w}^T \mathbf{z})^2\}$$

where $G$ is a non-linear function (typically $\log \cosh$ or exponential) that approximates negentropy. The resulting mixing matrix $A$ provides spatial maps indicating which features (neurons) participate in each independent component (functional network).

In the context of LLM analysis, ICA offers several advantages over alternative decomposition methods: (1) it identifies statistically independent components rather than merely uncorrelated ones (as in PCA), (2) it naturally handles the non-Gaussian activation distributions typical of neural networks, and (3) it has a well-established track record in neuroscience for identifying functional networks from neural signals.

### 2.5 The DOC Experimental Framework

Zhang et al. (2025) established a comprehensive experimental protocol for evaluating continual learning methods on LLMs that has become a standard benchmark in the field. Their framework includes:

**Datasets.** Fifteen text classification datasets spanning multiple domains:
- *Standard CL Benchmark* (5 datasets): AG News (topic), Amazon (sentiment), Yelp (sentiment), DBPedia (topic), Yahoo (topic)
- *GLUE tasks* (4 datasets): MNLI (NLI), QQP (paraphrase), RTE (NLI), SST-2 (sentiment)
- *SuperGLUE tasks* (5 datasets): WiC (word sense), CB (NLI), COPA (QA), BoolQ (boolean QA), MultiRC (multi-choice QA)
- *Additional*: IMDB (sentiment)

**Task Orders.** Six distinct task orderings to evaluate robustness:
- Orders 1–3: Standard benchmark with 5 tasks each (different permutations)
- Orders 4–6: Long-chain benchmark with 15 tasks each (mixing all dataset categories)

**Evaluation Metrics.** Three complementary metrics following Equation 15 of Zhang et al. (2025):
- *Average Accuracy (AA)*: $\text{AA}(T) = \frac{1}{T} \sum_{t=1}^{T} a_{t,T}$
- *Backward Transfer (BWT)*: $\text{BWT}(T) = \frac{1}{T-1} \sum_{t=1}^{T-1} (a_{t,T} - a_{t,t})$
- *Forward Transfer (FWT)*: $\text{FWT}(T) = \frac{1}{T-1} \sum_{t=2}^{T} (a_{t,t} - \tilde{a}_t)$

where $a_{t,T}$ denotes accuracy on task $t$ after training through task $T$, and $\tilde{a}_t$ is the baseline accuracy from single-task fine-tuning.

This protocol enables rigorous comparison across methods, controlling for task ordering effects and providing insight into both forgetting (BWT) and transfer (FWT) dynamics.

---

## 3. Our Contribution

While existing continual learning methods for LLMs operate primarily in the parameter space—constraining weight updates through regularization (EWC, LwF), orthogonal projection (O-LoRA), or dynamic direction tracking (DOC)—our approach operates in the *functional space* of the network. We identify and manipulate the functional networks that underlie the model's computational organization, providing a fundamentally different and complementary perspective on knowledge preservation.

**Neuroscience-Grounded Functional Network Identification.** Our primary contribution is the application of ICA-based functional network analysis to the continual learning setting. Rather than estimating parameter importance through local approximations (Fisher information) or tracking gradient directions (PCA/CCIPCA), we identify the intrinsic functional organization of the network through ICA decomposition of MLP activation patterns. This approach is grounded in decades of neuroscience research demonstrating that ICA reliably identifies functional networks in biological neural systems (Beckmann & Smith, 2004), and extends the recent finding that analogous networks exist in LLMs (Liu et al., 2025).

**Dual Masking Strategies.** We implement two complementary masking strategies that offer distinct approaches to knowledge preservation:

1. **Lesion mode**: Identifies functional networks associated with previously learned tasks and ablates them during new task training, preventing modification of critical representations. This is analogous to the neuroscience technique of lesion studies, where specific brain regions are deactivated to study their functional role.

2. **Preserve mode**: Restricts training to specific functional networks, isolating task-specific learning to designated network components. This approach leverages the finding from Liu et al. (2025) that approximately 10% of neurons within key functional networks are sufficient to maintain task performance.

**Template-Based Efficiency.** Unlike methods that require online computation during training (e.g., DOC's CCIPCA tracking), our approach pre-computes functional network templates that can be reused across multiple continual learning scenarios. Templates are computed once from a representative dataset and stored as lightweight JSON files containing per-component, per-layer channel indices. This separation of network identification from training enables efficient experimentation with different masking strategies and component selections.

**Anti-Drift Parametrization.** We introduce a row-wise parametrization mechanism (`RowMaskedDelta`) that prevents optimizer drift on frozen parameters. For masked neurons, the effective weight is computed as $W_{\text{eff}} = W_{\text{frozen}} + \mathbf{m} \odot \Delta$, where $\mathbf{m}$ is a binary row mask and $\Delta$ is a trainable delta initialized to zero. This ensures that frozen neurons maintain their exact pre-training values throughout the fine-tuning process, avoiding the subtle drift that can occur with standard gradient masking approaches.

**Comprehensive Evaluation Against State-of-the-Art.** We evaluate our approach using the complete DOC experimental protocol (Zhang et al., 2025), enabling direct comparison against five baseline methods across six task orders. This represents the most comprehensive evaluation of functional network-based continual learning to date.

---

## 4. Methodology

### 4.1 ICA-Based Functional Network Identification

Our method begins by identifying the functional networks within a pre-trained LLM through Independent Component Analysis of MLP activation patterns. This process follows the methodology established by Liu et al. (2025), adapted for the continual learning setting.

**Signal Capture.** We capture the final MLP outputs (post-activation, post down-projection) from all transformer layers during a forward pass over a representative dataset. For each input sample, we collect the activation tensor $Y_l \in \mathbb{R}^{T \times H}$ from each layer $l$, where $T$ is the sequence length and $H$ is the hidden dimension. Padding tokens are filtered using the attention mask to ensure only valid tokens contribute to the analysis.

**Design Matrix Construction.** The captured activations are concatenated across layers to form a global design matrix:

$$X = [Y_0, Y_1, \ldots, Y_{L-1}] \in \mathbb{R}^{T_{\text{total}} \times (L \cdot H)}$$

where $T_{\text{total}}$ is the total number of valid timesteps across all samples and $L$ is the number of layers. The design matrix is then standardized: $X_z = (X - \mu) / \sigma$, where $\sigma$ is clipped at $10^{-8}$ to prevent numerical instability.

**Dimensionality Reduction.** For large models where the design matrix exceeds LAPACK's integer limit ($2^{31} - 1$ elements) or the feature dimension exceeds a configurable threshold, PCA is applied as a preprocessing step:

$$X_{\text{reduced}} = X_z \cdot V_{\text{PCA}}$$

where $V_{\text{PCA}} \in \mathbb{R}^{(L \cdot H) \times K_{\text{PCA}}}$ contains the top $K_{\text{PCA}}$ principal components. ICA is then performed in this reduced space, and the resulting mixing matrix is projected back to the original feature space: $A = V_{\text{PCA}}^T \cdot A_{\text{reduced}}$.

**FastICA Decomposition.** We apply the FastICA algorithm (Hyvärinen & Oja, 2000) to the (optionally reduced) design matrix to obtain the mixing matrix $A \in \mathbb{R}^{(L \cdot H) \times K}$, where $K$ is the number of independent components. Each column $A_{:,c}$ represents the spatial map of component $c$, indicating the contribution of each neuron across all layers to that functional network.

**Component Mask Extraction.** For each component $c$, we extract a binary mask by thresholding the absolute values of the mixing matrix column:

$$\text{mask}_c(j) = \begin{cases} 1 & \text{if } |A_{j,c}| \geq \text{percentile}_p(|A_{:,c}|) \\ 0 & \text{otherwise} \end{cases}$$

where $p$ is a configurable percentile threshold (default: 98.0). The flat index $j$ is decomposed into layer and channel indices: $l = j \div H$, $\text{ch} = j \bmod H$. The resulting mask for each component is a dictionary mapping layer indices to lists of selected channel indices.

### 4.2 Masking Strategies

We implement two complementary masking strategies, applied via PyTorch forward hooks on MLP modules.

**Lesion Mode.** In lesion mode, selected functional networks are ablated by zeroing out their constituent neurons during the forward pass. For a given set of selected component IDs $\mathcal{C}$, the union of all channel indices across components is computed for each layer:

$$\text{channels}_l = \bigcup_{c \in \mathcal{C}} \text{mask}_c(l)$$

A binary mask tensor is constructed with ones everywhere except at the selected channels:

$$m_l[i] = \begin{cases} 0 & \text{if } i \in \text{channels}_l \\ 1 & \text{otherwise} \end{cases}$$

This mask is applied element-wise to the MLP output via a forward hook: $\hat{y}_l = y_l \odot m_l$, where $y_l \in \mathbb{R}^{B \times T \times H}$ is the MLP output and $m_l \in \mathbb{R}^{H}$ is broadcast across batch and sequence dimensions.

**Preserve Mode.** In preserve mode, only the selected functional networks are active; all other neurons are zeroed out:

$$m_l[i] = \begin{cases} 1 & \text{if } i \in \text{channels}_l \\ 0 & \text{otherwise} \end{cases}$$

This restricts the model's computation to the selected functional networks, isolating their contribution to the output.

**Gradient Implications.** In both modes, masked neurons (those multiplied by zero) receive zero gradients during backpropagation, effectively freezing them during training. This provides a hard constraint on which neurons can be modified, in contrast to the soft regularization of EWC or the subspace projection of O-LoRA.

### 4.3 Anti-Drift Parametrization

Standard gradient masking (zeroing gradients for frozen parameters) can still allow subtle parameter drift due to optimizer state accumulation (e.g., momentum in Adam). To address this, we introduce the `RowMaskedDelta` parametrization, which provides exact freezing guarantees.

For each MLP down-projection weight matrix $W \in \mathbb{R}^{H \times D_{\text{ff}}}$, we decompose the effective weight as:

$$W_{\text{eff}} = W_{\text{frozen}} + \mathbf{m} \odot \Delta$$

where:
- $W_{\text{frozen}}$ is the original weight matrix, stored as a non-trainable buffer
- $\mathbf{m} \in \{0, 1\}^{H \times 1}$ is a binary row mask (broadcast across columns)
- $\Delta \in \mathbb{R}^{H \times D_{\text{ff}}}$ is a trainable delta, initialized to zero

Rows where $\mathbf{m}[i] = 0$ are guaranteed to remain exactly at their frozen values regardless of optimizer behavior, since $W_{\text{eff}}[i] = W_{\text{frozen}}[i] + 0 \cdot \Delta[i] = W_{\text{frozen}}[i]$. Only rows where $\mathbf{m}[i] = 1$ receive gradient updates through $\Delta$.

The row mask is computed based on the masking mode:
- **Preserve mode**: Trainable rows correspond to selected component channels
- **Lesion mode**: Trainable rows correspond to the complement of selected component channels

This parametrization is applied to both LoRA B matrices (when using PEFT) and base weight matrices (when using full fine-tuning), and is removed ("baked") after training by setting the weight to its current effective value.

### 4.4 Template System

A key design decision in our approach is the separation of functional network identification from the training process. ICA templates are pre-computed once and stored as lightweight JSON files:

```json
{
  "name": "global_templates_v1",
  "layout": {
    "captured_layers_sorted": [0, 1, ..., L-1],
    "hidden_size": H
  },
  "templates": {
    "0": {"0": [ch_1, ch_2, ...], "1": [ch_3, ch_4, ...], ...},
    "1": {...},
    ...
  }
}
```

Each template contains: (1) metadata about the model architecture (layer ordering, hidden dimension), (2) per-component masks mapping layer indices to lists of selected channel indices. Templates are built from a representative dataset using the `build_ica_templates` tool, which supports multiple input datasets and configurable ICA parameters.

This template-based approach offers several advantages:
- **Reproducibility**: Identical masks are applied across experimental runs
- **Efficiency**: ICA computation (the most expensive step) is performed only once
- **Flexibility**: Different masking strategies and component selections can be explored without recomputing ICA
- **Portability**: Templates can be shared across research groups for reproducible comparisons

### 4.5 Continual Learning Integration

In the continual learning setting, our method operates as follows for a sequence of $T$ tasks:

**Algorithm: Functional Network Masking for Continual Learning**

1. **Pre-computation**: Build ICA templates from a representative dataset (or load pre-computed templates)
2. **For each task** $t = 1, 2, \ldots, T$:
   - a. **Before task**: If $t > 1$, select components to protect based on the component selection strategy and apply masks via forward hooks
   - b. **Train**: Fine-tune the model on task $t$ data using standard cross-entropy loss with LoRA adapters. Masked neurons receive zero gradients.
   - c. **After task**: Remove forward hooks. Optionally update the set of protected components.
3. **Evaluation**: After training on all tasks, evaluate on all task test sets to compute the accuracy matrix $A[t][T]$

**Component Selection Strategies.** We implement two strategies for selecting which components to protect:

1. **Cumulative**: After training on task $t$, protect components $\{0, 1, \ldots, t-1\}$. This progressively increases the number of protected networks as more tasks are learned.
2. **Fixed**: Protect a pre-specified set of component IDs throughout the entire training sequence. This allows manual selection of the most important functional networks based on prior analysis.

The cumulative strategy is the default, as it naturally scales protection with the number of learned tasks, analogous to how the brain might consolidate increasingly more neural circuits as experience accumulates.

---

## 5. Experimental Setup

### 5.1 Datasets and Task Orders

We adopt the complete experimental protocol from Zhang et al. (2025), using 15 text classification datasets organized into six task orders.

**Standard CL Benchmark.** Five datasets representing common text classification tasks:

| Dataset | Task Type | Classes | Source |
|---------|-----------|---------|--------|
| AG News | Topic classification | 4 | Zhang et al. (2015) |
| Amazon | Sentiment analysis | 2 | McAuley & Leskovec (2013) |
| Yelp | Sentiment analysis | 2 | Zhang et al. (2015) |
| DBPedia | Topic classification | 14 | Lehmann et al. (2015) |
| Yahoo | Topic classification | 10 | Zhang et al. (2015) |

**Extended Benchmark.** Ten additional datasets from GLUE, SuperGLUE, and IMDB:

| Dataset | Task Type | Classes | Source |
|---------|-----------|---------|--------|
| MNLI | Natural language inference | 3 | Williams et al. (2018) |
| QQP | Paraphrase detection | 2 | Iyer et al. (2017) |
| RTE | Natural language inference | 2 | Dagan et al. (2005) |
| SST-2 | Sentiment analysis | 2 | Socher et al. (2013) |
| WiC | Word sense disambiguation | 2 | Pilehvar & Camacho-Collados (2019) |
| CB | Natural language inference | 3 | De Marneffe et al. (2019) |
| COPA | Causal reasoning | 2 | Roemmele et al. (2011) |
| BoolQ | Boolean question answering | 2 | Clark et al. (2019) |
| MultiRC | Multi-sentence reading | 2 | Khashabi et al. (2018) |
| IMDB | Sentiment analysis | 2 | Maas et al. (2011) |

**Task Orders.** We evaluate across six task orders to assess robustness to ordering effects:

| Order | Type | Tasks |
|-------|------|-------|
| O1 | Standard (5 tasks) | AG News → Yelp → Amazon → DBPedia → Yahoo |
| O2 | Standard (5 tasks) | DBPedia → Yahoo → AG News → Amazon → Yelp |
| O3 | Standard (5 tasks) | Yelp → Yahoo → Amazon → DBPedia → AG News |
| O4 | Long chain (15 tasks) | Mixed ordering of all 15 datasets |
| O5 | Long chain (15 tasks) | Mixed ordering of all 15 datasets |
| O6 | Long chain (15 tasks) | Mixed ordering of all 15 datasets |

Each dataset is sampled to a maximum of 10,000 training examples per task, following the DOC protocol.

### 5.2 Baseline Methods

We compare our ICA-based functional network masking against five baseline methods:

| Method | Type | Key Mechanism | Reference |
|--------|------|---------------|-----------|
| LoRA | Vanilla baseline | Sequential LoRA fine-tuning, no CL mechanism | Hu et al. (2022) |
| EWC | Regularization | Fisher information matrix penalty ($\lambda = 0.4$) | Kirkpatrick et al. (2017) |
| LwF | Knowledge distillation | Soft target preservation ($\alpha = 1.0$, $T = 2.0$) | Li & Hoiem (2017) |
| O-LoRA | Orthogonal projection | SVD-based subspace orthogonality ($\lambda = 0.1$) | Wang et al. (2023) |
| DOC | Dynamic projection | Online PCA tracking + gradient projection ($\lambda = 0.5$) | Zhang et al. (2025) |

All methods use LoRA as the underlying parameter-efficient fine-tuning mechanism, ensuring a fair comparison of the continual learning strategies themselves.

### 5.3 Evaluation Metrics

We report three standard continual learning metrics:

**Average Accuracy (AA)** measures overall performance after training on all tasks:

$$\text{AA}(T) = \frac{1}{T} \sum_{t=1}^{T} a_{t,T}$$

**Backward Transfer (BWT)** quantifies catastrophic forgetting (negative values indicate forgetting):

$$\text{BWT}(T) = \frac{1}{T-1} \sum_{t=1}^{T-1} (a_{t,T} - a_{t,t})$$

**Forward Transfer (FWT)** measures the benefit of previous task learning on new tasks:

$$\text{FWT}(T) = \frac{1}{T-1} \sum_{t=2}^{T} (a_{t,t} - \tilde{a}_t)$$

where $a_{t,T}$ is the accuracy on task $t$ after training through task $T$, and $\tilde{a}_t$ is the single-task baseline accuracy. Results are reported as averages across the three standard orders (O1–O3) and three long-chain orders (O4–O6).

### 5.4 Implementation Details

**Model.** We use Llama-3.2-1B-Instruct (Meta, 2024) as our primary model, a 1-billion parameter instruction-tuned language model. This model provides a balance between computational efficiency and representational capacity suitable for extensive continual learning experiments.

**LoRA Configuration.**

| Parameter | Value |
|-----------|-------|
| LoRA rank ($r$) | 16 |
| LoRA alpha ($\alpha$) | 32 |
| LoRA dropout | 0.05 |
| Target modules | `down_proj`, `up_proj` |

**Training Hyperparameters.**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-4}$ |
| Batch size | 8 |
| Training steps per task | 1,000 |
| Gradient clipping | 1.0 (max norm) |
| Warmup | Linear warmup schedule |
| Precision | float32 (MPS) / bfloat16 (CUDA) |

**ICA Template Parameters.**

| Parameter | Value |
|-----------|-------|
| Number of components ($K$) | 10 |
| Percentile threshold ($p$) | 98.0 |
| Sample batches | 100 |
| Max PCA components | 1,000 |
| ICA algorithm | FastICA (max_iter=200, tol=1e-3) |
| Random seed | 0 (for reproducibility) |

**ICA Masking Configuration.**

| Parameter | Value |
|-----------|-------|
| Mask mode | Lesion / Preserve (evaluated separately) |
| Component selection | Cumulative |
| Anti-drift parametrization | Enabled |
| Hook target | MLP down-projection output |

**Computational Resources.** Experiments are conducted on [hardware specification to be added]. ICA template computation requires approximately [time to be measured] per model. Each continual learning run (one task order, one method) requires approximately [time to be measured].

### 5.5 Comparison with DOC Paper Protocol

Our experimental setup closely follows the DOC paper (Zhang et al., 2025) with the following correspondences and differences:

| Aspect | DOC Paper | Our Setup |
|--------|-----------|-----------|
| Model | LLaMA-7B, LLaMA-13B, T5-Large | Llama-3.2-1B-Instruct |
| Task orders | 6 (O1–O6) | 6 (O1–O6, identical) |
| Datasets | 15 | 15 (identical) |
| Training steps | 1,000 per task | 1,000 per task |
| LoRA rank | 16, 64 | 16 |
| Metrics | AA, BWT, FWT | AA, BWT, FWT |
| Baselines | LoRA, EWC, LwF, O-LoRA | LoRA, EWC, LwF, O-LoRA, DOC |

The primary difference is our use of Llama-3.2-1B-Instruct rather than the larger LLaMA-7B/13B models used in the DOC paper. This choice balances computational feasibility with model capability, and we note that the DOC paper demonstrated consistent relative performance rankings across model sizes.

---

## 6. Results

> **[PLACEHOLDER — Experiments Ongoing]**
>
> This section will present the experimental results upon completion of the evaluation protocol. The following tables and analyses are planned:
>
> **Table 1: Average Accuracy (AA) — Standard CL Benchmark**
> Comparison of all methods across task orders O1–O3, reporting per-order AA and the average across standard orders.
>
> | Method | O1 | O2 | O3 | Std Avg |
> |--------|----|----|----|---------|
> | LoRA | — | — | — | — |
> | EWC | — | — | — | — |
> | LwF | — | — | — | — |
> | O-LoRA | — | — | — | — |
> | DOC | — | — | — | — |
> | **ICA-FNM (Lesion)** | — | — | — | — |
> | **ICA-FNM (Preserve)** | — | — | — | — |
>
> **Table 2: Average Accuracy (AA) — Long Chain Benchmark**
> Comparison of all methods across task orders O4–O6, reporting per-order AA and the average across long-chain orders.
>
> | Method | O4 | O5 | O6 | Long Avg |
> |--------|----|----|----|---------|
> | LoRA | — | — | — | — |
> | EWC | — | — | — | — |
> | LwF | — | — | — | — |
> | O-LoRA | — | — | — | — |
> | DOC | — | — | — | — |
> | **ICA-FNM (Lesion)** | — | — | — | — |
> | **ICA-FNM (Preserve)** | — | — | — | — |
>
> **Table 3: Backward Transfer (BWT) and Forward Transfer (FWT)**
> Forgetting and transfer metrics for all methods, averaged across standard and long-chain orders.
>
> | Method | Std BWT | Std FWT | Long BWT | Long FWT |
> |--------|---------|---------|----------|----------|
> | LoRA | — | — | — | — |
> | EWC | — | — | — | — |
> | LwF | — | — | — | — |
> | O-LoRA | — | — | — | — |
> | DOC | — | — | — | — |
> | **ICA-FNM (Lesion)** | — | — | — | — |
> | **ICA-FNM (Preserve)** | — | — | — | — |
>
> **Additional Planned Analyses:**
> - Per-task accuracy breakdown showing which tasks benefit most from ICA masking
> - Ablation study on the number of ICA components ($K$) and percentile threshold ($p$)
> - Analysis of component overlap across tasks (do different tasks activate different functional networks?)
> - Comparison of cumulative vs. fixed component selection strategies
> - Computational overhead analysis (wall-clock time per training step)
> - Visualization of functional network spatial patterns across layers

---

## 7. Discussion

> **[PLACEHOLDER — To Be Completed After Results]**
>
> The discussion section will address the following topics based on experimental outcomes:
>
> **7.1 Effectiveness of Functional Network Masking**
> - How does ICA-based masking compare to parameter-space methods (EWC, O-LoRA, DOC)?
> - Does the neuroscience-inspired approach provide advantages in specific scenarios (e.g., long task chains, diverse task types)?
> - What is the trade-off between knowledge preservation (BWT) and new task learning (FWT)?
>
> **7.2 Lesion vs. Preserve Mode Analysis**
> - Which masking strategy is more effective for continual learning?
> - How does the choice of mode interact with the number of components and task ordering?
> - Are there task-specific patterns in which mode performs better?
>
> **7.3 Interpretability and Functional Organization**
> - What do the identified functional networks reveal about the organization of knowledge in LLMs?
> - Do different tasks activate distinct functional networks, or is there significant overlap?
> - How do functional network patterns compare across layers (shallow vs. deep)?
>
> **7.4 Comparison with DOC's Functional Direction Tracking**
> - How does our static template-based approach compare to DOC's dynamic tracking?
> - What are the trade-offs between pre-computed templates and online adaptation?
> - Could the two approaches be combined for improved performance?
>
> **7.5 Limitations and Future Directions**
> - Scalability to larger models (7B, 13B, 70B parameters)
> - Extension to non-classification tasks (generation, reasoning, code)
> - Dynamic template updating during continual learning
> - Integration with other PEFT methods beyond LoRA
> - Theoretical analysis of the relationship between ICA components and task-specific knowledge

---

## 8. Conclusion

> **[PLACEHOLDER — To Be Completed After Results]**
>
> The conclusion will summarize:
> 1. The key findings from our experimental evaluation
> 2. The practical implications of functional network masking for continual learning in LLMs
> 3. The broader significance of neuroscience-inspired approaches to understanding and improving LLM training
> 4. Concrete directions for future research building on this work
>
> We anticipate concluding that ICA-based functional network masking provides a novel and interpretable approach to continual learning that complements existing parameter-space methods, with the potential to offer unique insights into the functional organization of knowledge within large language models.

---

## References

Beckmann, C. F., & Smith, S. M. (2004). Probabilistic independent component analysis for functional magnetic resonance imaging. *IEEE Transactions on Medical Imaging*, 23(2), 137–152.

Biswal, B., Yetkin, F. Z., Haughton, V. M., & Hyde, J. S. (1995). Functional connectivity in the motor cortex of resting human brain using echo-planar MRI. *Magnetic Resonance in Medicine*, 34(4), 537–541.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901.

Calhoun, V. D., Adali, T., Pearlson, G. D., & Pekar, J. J. (2001). A method for making group inferences from functional MRI data using independent component analysis. *Human Brain Mapping*, 14(3), 140–151.

Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019). Efficient lifelong learning with A-GEM. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., & Toutanova, K. (2019). BoolQ: Exploring the surprising difficulty of natural yes/no questions. *Proceedings of NAACL-HLT*, 2924–2936.

Dagan, I., Glickman, O., & Magnini, B. (2005). The PASCAL recognising textual entailment challenge. *Machine Learning Challenges Workshop*, 177–190.

De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., ... & Tuytelaars, T. (2021). A continual learning survey: Defying forgetting in classification tasks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(7), 3366–3385.

De Marneffe, M.-C., Simons, M., & Tonhauser, J. (2019). The CommitmentBank: Investigating projection in naturally occurring discourse. *Proceedings of Sinn und Bedeutung*, 23(2), 107–124.

French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128–135.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Hyvärinen, A., & Oja, E. (2000). Independent component analysis: Algorithms and applications. *Neural Networks*, 13(4–5), 411–430.

Hyvärinen, A., & Pajunen, P. (1999). Nonlinear independent component analysis: Existence and uniqueness results. *Neural Networks*, 12(3), 429–439.

Iyer, S., Dandekar, N., & Csernai, K. (2017). First Quora dataset release: Question pairs. *Quora Blog*.

Khashabi, D., Chaturvedi, S., Roth, M., Upadhyay, S., & Roth, D. (2018). Looking beyond the surface: A challenge set for reading comprehension over multiple sentences. *Proceedings of NAACL-HLT*, 252–262.

Khemakhem, I., Kingma, D., Monti, R., & Hyvärinen, A. (2020). Variational autoencoders and nonlinear ICA: A unifying framework. *Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2207–2217.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.

Le, Q. V., Karpenko, A., Ngiam, J., & Ng, A. Y. (2011). ICA with reconstruction cost for efficient overcomplete feature learning. *Advances in Neural Information Processing Systems*, 24, 1017–1025.

Lehmann, J., Isele, R., Jakob, M., Jentzsch, A., Kontokostas, D., Mendes, P. N., ... & Bizer, C. (2015). DBpedia—A large-scale, multilingual knowledge base extracted from Wikipedia. *Semantic Web*, 6(2), 167–195.

Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(12), 2935–2947.

Liu, Z., Li, X., Zhu, H., Liu, Z., & Sun, M. (2025). Brain-inspired exploration of functional networks and key neurons in large language models. *arXiv preprint arXiv:2501.xxxxx*.

Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *Advances in Neural Information Processing Systems*, 30, 6467–6476.

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics*, 142–150.

Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7765–7773.

McAuley, J., & Leskovec, J. (2013). Hidden factors and hidden topics: Understanding rating dimensions with review text. *Proceedings of the 7th ACM Conference on Recommender Systems*, 165–172.

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109–165.

Meta. (2024). Llama 3.2: Revolutionizing edge AI and vision with open, customizable models. *Meta AI Blog*.

Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*, 113, 54–71.

Pilehvar, M. T., & Camacho-Collados, J. (2019). WiC: The word-in-context dataset for evaluating context-sensitive meaning representations. *Proceedings of NAACL-HLT*, 1267–1273.

Power, J. D., Cohen, A. L., Nelson, S. M., Wig, G. S., Barnes, K. A., Church, J. A., ... & Petersen, S. E. (2011). Functional network organization of the human brain. *Neuron*, 72(4), 665–678.

Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review*, 97(2), 285–308.

Roemmele, M., Bejan, C. A., & Gordon, A. S. (2011). Choice of plausible alternatives: An evaluation of commonsense causal reasoning. *AAAI Spring Symposium: Logical Formalizations of Commonsense Reasoning*.

Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soez, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016). Progressive neural networks. *arXiv preprint arXiv:1606.04671*.

Smith, S. M., Fox, P. T., Miller, K. L., Glahn, D. C., Fox, P. M., Mackay, C. E., ... & Beckmann, C. F. (2009). Correspondence of the brain's functional architecture during activation and rest. *Proceedings of the National Academy of Sciences*, 106(31), 13040–13045.

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, 1631–1642.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.

Wang, Q., Fink, O., Van Gool, L., & Dai, D. (2023). Orthogonal subspace learning for language model continual learning. *Findings of the Association for Computational Linguistics: EMNLP 2023*, 10658–10671.

Williams, A., Nangia, N., & Bowman, S. (2018). A broad-coverage challenge corpus for sentence understanding through inference. *Proceedings of NAACL-HLT*, 1112–1122.

Yeo, B. T. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D., Hollinshead, M., ... & Buckner, R. L. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106(3), 1125–1165.

Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 3987–3995.

Zhang, X., Peng, H., Li, H., Song, L., & Li, Y. (2025). Dynamic orthogonal continual fine-tuning for mitigating catastrophic forgetting. *arXiv preprint arXiv:2501.xxxxx*.

Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *Advances in Neural Information Processing Systems*, 28, 649–657.