---
title: "Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models"
authors: ["Yiheng Liu", "Xiaohui Gao", "Haiyang Sun", "Bao Ge", "Tianming Liu", "Junwei Han", "Xintao Hu"]
venue: "ICML 2025 (PMLR 267)"
arxiv: "2502.20408v1"
date: "2025-02-13"
code: "WhatAboutMyStar/LLM_ACTIVATION"
---

# Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models

**Authors:** Yiheng Liu; Xiaohui Gao; Haiyang Sun; Bao Ge; Tianming Liu; Junwei Han; Xintao Hu
<!-- page:1 -->
## Abstract
In recent years, the rapid advancement of large
language models (LLMs) in natural language processing has sparked significant interest among
researchers to understand their mechanisms and
functional characteristics. Although existing studies have attempted to explain LLM functionalities
by identifying and interpreting specific neurons,
these efforts mostly focus on individual neuron
contributions, neglecting the fact that human brain
functions are realized through intricate interaction
networks. Inspired by cognitive neuroscience research on functional brain networks (FBNs), this
study introduces a novel approach to investigate
whether similar functional networks exist within
LLMs. We use methods similar to those in the
field of functional neuroimaging analysis to locate
and identify functional networks in LLM. Experimental results show that, similar to the human
brain, LLMs contain functional networks that frequently recur during operation. Further analysis
shows that these functional networks are crucial
for LLM performance. Masking key functional
networks significantly impairs the model's performance, while retaining just a subset of these
networks is adequate to maintain effective operation. This research provides novel insights into
the interpretation of LLMs and the lightweighting of LLMs for certain downstream tasks.
Code is available at https://github.com/
WhatAboutMyStar/LLM_ACTIVATION.
*Equal contribution 1School of Automation, Northwestern Polytechnical University, Xi'an, China 2School of Physics and Information Technology, Shaanxi Normal University, Xi'an, China 3School
of Computing, University of Georgia, Athens, USA. Correspondence to: Xintao Hu <xhu@nwpu.edu.cn>.
Proceedings of the 41 st International Conference on Machine
Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025
by the author(s).
## 1. Introduction
In recent years, large language models (LLMs) have become
a focal point of research in the field of artificial intelligence
(AI) due to their remarkable capabilities in natural language
processing (Zhao et al., 2024a; 2023; Liu et al., 2023b;
Wang et al., 2024; Liu et al., 2025). However, these models
are often considered "black boxes", with insufficient understanding of their internal mechanisms, which limits the
potential for further optimization and broader application.
Therefore, exploring methods to explain and understand
LLMs is essential both for improving model transparency
and trustworthiness and for establishing a foundation to
develop more efficient and reliable AI systems.
One research direction on the mechanistic interpretability
of LLMs focuses on the analysis of individual neurons (Yu
& Ananiadou, 2024a; Dai et al., 2022). A subset of this
research is dedicated to the identification of neurons that are
crucial to the functionality of LLMs, with the aim of locating and assessing the functions of these key neurons (Yu &
Ananiadou, 2024b; Niu et al., 2024; Chen et al., 2024). Studies have shown that the removal of certain neurons leads to a
significant degradation in performance in LLMs, highlighting their essential role in maintaining the core functions of
the model. In addition, another line of research investigates
the specific functional roles of these neurons (AlKhamissi
et al., 2024; Wang et al., 2022). For example, some neurons
may specialize in processing linguistic structures, while others might be responsible for reasoning tasks (Huo et al.,
2024; Zhao et al., 2024b). Furthermore, by manipulating
these key neurons, such as amplifying or masking their outputs (Song et al., 2024), researchers have demonstrated the
ability to control and predict the behavior of LLMs. Methods for identifying important neurons within LLM can be
categorized into several approaches. These include analyzing the gradients of neurons to evaluate their impact on
model predictions (Sundararajan et al., 2017; Lundstrom
et al., 2022), employing causal tracing techniques to uncover the causal relationships that influence model behavior
(Nikankin et al., 2024), and conducting statistical analyses
of activated neurons to measure their information content
and variability (AlKhamissi et al., 2024; Song et al., 2024;
Tang et al., 2024). These approaches provide valuable tools
1

<!-- page:2 -->
for understanding and explaining LLMs, offering deeper
insight into their inner mechanisms.
However, the function of an individual neuron is much more
complex than it might initially seem. A single neuron is
not only dedicated to performing a single task, but can simultaneously participate in multiple functional modules
(Mountcastle, 1997; Miller & Cohen, 2001; Raichle & Snyder, 2007). Neurons often form functional networks through
their interactions and connectivity, collaboratively working
to perform higher-level cognitive tasks (Smith et al., 2009;
Bullmore & Sporns, 2009). The role of a neuron therefore extends beyond its individual activation patterns and is
shaped by its cooperation with other neurons within these
networks (Bullmore & Sporns, 2009; Liu et al., 2024b;a).
Existing research has focused mainly on the parameters
and structure of neurons, primarily examining their individual properties and connectivity. These approaches, which
emphasize structural and parametric analysis, overlook the
functional network perspective and fail to explore the roles
and contributions of neurons within these networks. As a
result, these limitations have hindered a deeper understanding of neuronal function, neglecting the insights offered
by neuroscience research on FBNs (Hassabis et al., 2017;
Vilas et al.). To address these challenges, we draw inspiration from cognitive neuroscience to investigate whether
LLMs contain functional networks similar to those found in
the human brain. By recognizing the similarities between
functional magnetic resonance imaging (fMRI) (Matthews
& Jezzard, 2004; Logothetis, 2008) signals and the output
signals of neurons in LLM, we hypothesized that the techniques used in fMRI analysis could be adapted to analyze
LLM neurons. Specifically, we treated the neuron outputs
from the multilayer perception (MLP) layers of LLMs as
analogous to fMRI signals and applied Independent Component Analysis (ICA) (Hyv¨arinen & Oja, 2000; Beckmann
et al., 2005; Varoquaux et al., 2010b) to decompose these
neuron outputs into multiple functional networks.
Our experiments on extensive datasets confirmed the existence of numerous functional networks within LLMs. Just
as we derived functional networks from fMRI signals (Mensch et al., 2016; Varoquaux et al., 2010a; Liu et al., 2023a;
He et al., 2023; Ge et al., 2020; Lv et al., 2015). Some of the
functional networks exhibit high spatial consistency across
various input stimuli and play a crucial role in the model's
functionality. We discovered that masking specific key networks, which typically consist of less than 2% neurons, can
significantly impair model performance. Conversely, maintaining these essential networks while gradually integrating
additional decomposed functional networks and masking
non-essential neurons enables the model's performance to
improve progressively from low to high. Ultimately, by
utilizing less than one-tenth of the neurons in the MLP layer,
we achieved performance that matches that of the original
network.
Our contributions can be summarized as follows:
- Bridging Cognitive Neuroscience and AI: We have introduced methods from cognitive neuroscience to analyze neurons in LLM, creating a link between brain science and AI. This integration offers new research directions for braininspired AI, providing fresh insights into how neural networks operate.
- Discovery of Functional Networks in LLMs: We confirmed that LLMs contain functional networks similar to those found in the human brain. Some of these functional networks exhibit high spatial consistency in various input stimuli, demonstrating relatively stable spatial patterns. This finding suggests that LLMs may share fundamental organizational principles with biological brains.
- Validation of Key Neurons in Functional Networks: We have demonstrated that neurons within these functional networks are essential for maintaining the functionality of LLMs. These neurons play a crucial role in ensuring that the model operates effectively, underscoring their importance for the overall performance of the system.

## 2. Preliminaries
In this section, we provide the essential background knowledge required for understanding our work, including the
MLP layer in LLMs, how neuroscience utilizes fMRI to
study brain functions, and the ICA algorithm which we
use to decompose neural activity into distinct functional
networks.
Transformer, MLP layer: LLMs utilized in this paper
are based on the transformer architecture (Vaswani, 2017),
specifically employing a decoder-only configuration. (Radford et al., 2018; 2019; Brown et al., 2020; Yang et al.,
2024b; GLM et al., 2024). In this configuration, each transformer decoder consists of two primary components: a
multi-head self-attention module and a MLP module. The
non-linear transformations in transformer models occur
within the MLP layers. Typically, an MLP module consists of two fully connected layers. The first layer increases
the dimensionality, often to four times the original dimension, followed by a non-linear activation function such as the
Sigmoid-weight Linear Unit (SiLU) (Elfwing et al., 2018)
or the Gaussian Error Linear Unit (GELU) (Hendrycks &
Gimpel, 2016). The second layer then reduces the dimensionality back to its original size. In our study, we focus on
the neurons located in the final MLP layer of each decoder
module within the model. Given an input vector x ∈Rdmodel,
the MLP module can be represented as follows:
MLP(x) = W2 · σ(W1 · x + b1) + b2
(1)

<!-- page:3 -->

> **Figure 1.** The total framework of the paper. Functional networks generated by ICA on fMRI signals and LLM's neurons are shown in
upper and lower part of the figure respectively.

W is the weight matrix of the linear transformation, b is the
bias vector of the linear transformation, σ is a non-linear activation function. The neuron outputs used in this paper are
the outputs of the MLP(x). This choice allows us to analyze
the impact of these neurons on the final output, as they are
directly responsible for the refined feature extraction and
transformation before the model generates its predictions.
Functional Brain Networks, FBNs: FBNs refer to collections of brain regions that are co-activated during specific
tasks or while at rest (Dong et al., 2020). FMRI is a noninvasive technique used to measure Blood-Oxygen-Level
Dependent (BOLD) signals, which reflect neuronal activity
indirectly (Matthews & Jezzard, 2004; Logothetis, 2008).
The intensity of voxel values in fMRI signals indirectly reflects neuronal activity by capturing variations in local blood
oxygen levels due to neural metabolism. Neuroscientific research hypothesis that the observed BOLD signals in fMRI
signals are likely the result of multiple independent functional networks working together. Essentially, these BOLD
signals can be considered as linear combinations of several source signals, each representing a distinct functional
network. By comparing the spatial patterns of functional
networks across various task conditions, researchers can
infer which regions of brain are associated with specific
cognitive or behavioral functions.
Independent Component Analysis, ICA: ICA is a powerful data-driven technique used to extract source signals that
are as statistically independent as possible from a mixed
signal. Within the field of neuroscience, ICA is frequently
applied to fMRI data to uncover underlying functional networks, offering valuable insights into how different regions
of the brain collaborate to facilitate cognitive and behavioral processes. ICA disentangles mixed fMRI signals into
several independent components, where each component
represents a distinct functional network. Each extracted
independent component is associated with a spatial map that
illustrates which brain regions contribute to that component.
These contributing regions typically display synchronized
activity patterns, indicating their coordinated involvement
in specific neural processes.
The objective of ICA is to recover the source signals S from
the observed signals X. Suppose that we have n observed
signals [x1, x2, . . . , xn], which are linear mixtures of m independent source signals [s1, s2, . . . , sm]. The relationship

<!-- page:4 -->
between the observed signals X and the source signals S
can be expressed as:
X = AS
(2)
where A is the mixing matrix that describes how the source
signals are combined to produce the observed signals. Each
row of the linear mixing matrix represents the spatial pattern
of the corresponding functional network, which illustrates
specific regions are activated. In this study, the functional
networks derived from LLMs neuron signals refer to the
rows of the linear mixing matrix A, which indicate the set
of neurons that are consistently co-activated under different
conditions. FastICA (Hyv¨arinen & Oja, 2000) is an efficient
algorithm for implementing ICA and is the method used
in this paper to derive FBNs from LLMs. The FastICA
algorithm can be described as follows:
Pre-whitening: The signals are first centered (zero mean)
and whitened to remove any linear correlations between the
variables and have unit variance.
The whitened signals Z can be represented as:
Z = E-1/2V-1(X -E[X])
(3)
where V and E are the eigenvectors and eigenvalues of the
covariance matrix Σ of X.
Finding Independent Components: For each independent
component wi, we maximize the following objective function:
J(w) = [E{G(wT z)}] -1
2E{(wT z)2}
(4)
where G is a non-linear function used to approximate negentropy, a1 is a constant usually a1 ∈[1, 2], Common choices
for G include:
G(u) = log cosh(a1u)
(5)
G(u) = -exp(-u2/2)
(6)
To find the optimal w, FastICA uses fixed-point iteration:
wnew = E{zg(wT z)} -E{g′(wT z)}w
(7)
where g is the derivative of G. After each iteration, normalize w:
w ←
w
∥w∥
(8)
If multiple independent components need to be extracted,
perform orthogonalization to ensure that the weight vectors
remain orthogonal:
W ←(WWT )-1/2W
(9)
where W is the matrix that contains all weight vectors
as columns. Repeat the iteration until the change in the
weight vectors falls below a predefined threshold, indicating
convergence. Once the demixing matrix W is obtained, the
source signals S can be estimated from the whitened data
Z:
ˆS = WZ
(10)
Finally, the mixing matrix A, which represents the spatial
patterns of the functional networks, can be obtained as the
inverse of the matrix W. Considering the whitening transformation applied to the signals, the mixing matrix can be
computed as:
A = VE1/2W-1
(11)
In visualizations and experimental comparisons, the mixing
matrix A, representing the final derived functional networks,
typically undergoes thresholding. This process involves
setting a threshold to filter out lower values, ensuring that
only the regions with significant activation are retained. This
approach helps in focusing on the most relevant activations
and improving the clarity of the results.
## 3. Functional Networks in Large Language
Models
In this section, we use the ICA to explore functional networks within LLMs.
### 3.1. Datasets and Models
We used text from four different types of datasets to validate the differences in functional networks when the model
performs various tasks. Specifically, we used news articles
from the AGNEWS dataset (Zhang et al., 2015), encyclopedia entries from Wikitext2 (Merity et al., 2016), mathematical texts from MathQA (Amini et al., 2019), and code
snippets from the CodeNet dataset (Puri et al., 2021). Additionally, we conducted experiments using three different
LLMs: GPT2 (Radford et al., 2019), Qwen2.5-3B (Yang
et al., 2024a), and ChatGLM3-6B-base (GLM et al., 2024).
### 3.2. Extracting Functional Networks from LLMs
The general workflow framework of our work is shown
in Figure 1. To extract functional networks from LLMs,
the first step is to determine which neurons' output signals
to use. In this paper, we utilize the neurons from the last
layer of each model's MLP module, specifically the neuron
output from these MLP modules. We collect the neuron
output from all MLP modules in the model.

<!-- page:5 -->
**Table 1. Statistics of Functional Networks Similar to the Template**
|TEMPLATES|NEWS|WIKI|MATH|CODE|
|---|---|---|---|---|
|1|15|208|142|140|
|2|7|119|109|85|
|3|12|59|31|44|
|4|31|145|91|114|
|5|32|91|32|135|
|6|87|4|6|107|
|7|63|41|35|134|
|8|21|58|43|71|
|9|15|0|51|112|
|10|11|18|118|79|
In neuroscience research, functional networks are typically
derived from fMRI signals, which reflect neural activity in
response to various stimuli. Interestingly, there is a notable
similarity between the output signals of neurons in LLMs
and fMRI signals. Both LLM neuron outputs and fMRI
signals not only capture neural activity but also exhibit temporal characteristics in response to different inputs. This
similarity suggests the potential to apply analytical techniques used for fMRI signals to LLMs. By leveraging these
parallels, we can explore and analyze functional networks
within LLMs in a manner similar to our analysis of human
brain data, thus opening a new way of understanding the
internal mechanisms of these models.
fMRI analysis typically involves performing ICA on individual data or performing group-wise analysis on data from
multiple individuals. In individual analysis, ICA is applied
to single-subject fMRI data. This method provides detailed
insights into the unique characteristics of an individual's
brain activity patterns. In contrast, in group-wise analysis,
ICA is performed on data from multiple subjects. This approach improves the generalizability of the research findings.
For group-wise analysis, a common approach is to stack the
data of all subjects together to form a large data matrix and
then perform ICA on this combined dataset. We randomly
selected 100 samples from each dataset for group-wise ICA
analysis, obtaining 10 functional network templates. We
then randomly selected 100 additional samples for individual analysis.
The functional networks derived from the group-wise ICA
analysis can be considered as templates representing a generalization of functionality. We computed the spatial similarity between these templates and the functional networks
derived from the individual analysis of another 100 samples.
In neuroscience research, the intersection over union (IoU)
is commonly used as a metric to evaluate the similarity between functional networks. We quantify spatial similarity
using the IoU between two functional networks N (1) and
N (2). Here, n represents the number of neurons, and the
IoU is defined as follows:
**Table 2. Statistics of Functional Networks Similar to the Template**
|TEMPLATES|NEWS|WIKI|MATH|CODE|
|---|---|---|---|---|
|1|0|2|0|0|
|2|0|0|1|59|
|3|0|1|0|0|
|4|0|0|0|0|
|5|50|10|0|0|
|6|0|17|0|0|
|7|64|73|55|58|
|8|0|0|0|64|
|9|0|0|0|3|
|10|64|70|55|2|
**Table 3. Statistics of Functional Networks Similar to the Template**
|TEMPLATES|NEWS|WIKI|MATH|CODE|
|---|---|---|---|---|
|1|0|0|0|0|
|2|1|0|0|0|
|3|0|0|0|0|
|4|0|0|0|0|
|5|0|0|0|2|
|6|3|0|8|0|
|7|0|0|1|0|
|8|0|0|0|0|
|9|0|4|0|0|
|10|0|0|0|0|
IoU(N (1), N (2)) =
Pn
i=1 |N (1)
i
∩N (2)
i
|
Pn
i=1 |N (1)
i
∪N (2)
i
|
(12)
Tables
1,
2, and
3 present the number of functional
networks derived from individual analysis with a spatial
similarity greater than 0.2 compared to the template, for
GPT-2, Qwen2.5-3B-Instruct, and ChatGLM3-6B-base, respectively. According to our experimental findings and
experience in neuroscience research, when different functional networks have spatial similarity greater than 0.2, they
appear reasonably similar on subjective visual inspection.
Consequently, such networks are classified into the same
category of functional networks.
We observe that, even when the inputs are different samples,
there are still a significant number of functional networks
similar to the templates. This indicates that LLMs indeed
contain specific functional patterns similar to those found
in human brain. Interestingly, we noticed that as model
size increases, it becomes progressively more challenging
to find functional networks similar to the templates within
individual samples. We do not interpret this as larger models that have fewer or no specific spatial pattern functional
networks. Instead, we believe that as model size grows,
the functionalities are divided into finer components that in5

<!-- page:6 -->
volve more functional networks. When dealing with shorter
texts, decomposition into only 10 components may not be
sufficient to achieve the same level of granularity as the template functional networks. Consequently, a greater number
of decomposed network components are required to capture these finer details; this assumption is contingent upon
having sufficiently long time series data for adequate decomposition. With sufficient data, a larger number of functional
networks similar to the templates can be identified.
> **Figure 2.** Some similar functional networks in templates. In the
figure, each row represents the neurons in a MLP layer, with
neurons being highlighted if they are activated.
### 3.3. Visualiztion of Functional Networks
In Figure 2, we present several functional networks derived
from the data. These networks involve a very low percentage of total neurons, which aligns with our understanding
that neurons in LLMs are sparsely activated. Additionally,
the figure illustrates that similar functional networks can be
obtained from different types of input data. This suggests
that certain LLM functional networks are broadly presented
and engaged in specific tasks, this is similar to the functional
networks in the human brain. For instance, the default mode
network (DMN) in the human brain is known to be involved
in various cognitive tasks (Raichle, 2015). Furthermore,
our observations indicate that the majority of neurons activated within functional networks are located in the deeper
layers of the models. This finding implies that functional
connectivity among neurons is more pronounced in these
deeper layers. For a more comprehensive examination of
these functional network patterns, please refer to our supplementary material.
**Table 4. Performance of GPT-2 with Masked Functional Networks**
on the AGNews Dataset. The first column represents each functional network and the number of neurons that are masked. The
second column shows the accuracy. The model's accuracy under
normal conditions (without any masking) is 0.8614.
FUNCTIONAL NETWORKS (9216)
ACCURACY
1→37
0.8538
2→21
0.6643
3→57
0.8391
4→29
0.8439
5→36
0.5749
6→29
0.8500
7→15
0.8482
8→55
0.7042
9→15
0.6753
10→72
0.8471
## 4. Evaluating the Importance of Functional
Networks
In this section, we will evaluate the functional networks
obtained using ICA to verify whether they play a critical
role in the performance of LLMs.
### 4.1. Datasets and Evaluate Criteria
In this section, we continue to use the GPT-2, Qwen2.53B-Instruct, and ChatGLM3-6B-base for evaluation. To
assess the model's performance, we used datasets from
the General Language Understanding Evaluation (GLUE)
(Wang, 2018), the Stanford Question Answering Dataset
(SQuAD) (Rajpurkar et al., 2016), and AGNews (Zhang
et al., 2015).
We fine-tuned GPT-2 in these datasets to adapt it to perform
specific tasks. In contrast, for Qwen2.5-3B-Instruct and
ChatGLM3-6B-base, we utilized carefully crafted prompts
to evaluate their performance in a zero-shot setting on the
same datasets. In our assessments, we utilized accuracy
as the performance metric for the GLUE and AGNews
datasets. For the SQuAD, which involves more complex
question-answering tasks, we used the F1 score to evaluate
the model's performance. The underlying hypothesis is that
if these neurons play a critical role in the model's functionality, then ensuring their activation alone should be enough
to maintain the model's performance.
### 4.2. Neuron Lesion Experiment
In the neuron lesion experiment, we deactivate functional
networks within the LLMs. The goal is to investigate how
the removal or deactivation of these neurons impacts the
overall performance and functionality of the model.
Table 4 presents the performance results of the GPT-2 model
in the AGNews dataset after selectively masking specific

<!-- page:7 -->
**Table 5. Performance of GPT-2 with Masked Functional Networks.**
First Column: Dataset names. Second Column: Model performance under normal conditions (without any masking). Third
Column: Model performance after masking 15% of the neurons
randomly. Fourth Column: Model performance after masking the
neurons belonging to 10 specific functional networks (The number
of neurons that are masked is less than 2% of the total neurons.).
DATASETS
NORMAL
MASKED 15%
MASKED 10
COLA
0.7776
0.7383
0.5772
MRPC
0.7941
0.7941
0.7377
QQP
0.8758
0.8717
0.8104
SST-2
0.9163
0.9083
0.5436
MNLI
0.6847
0.6879
0.5995
QNLI
0.8746
0.8569
0.6811
AG NEWS
0.8614
0.8487
0.5557
SQUAD
0.3397
0.3154
0.1342
**Table 6. Performance of Qwen2.5-3B-Instruct with Masked Functional Networks.**
|DATASETS|QWEN2.5|NORMAL|QWEN2.5|MASKED|
|---|---|---|---|---|
|10|COLA|0.6913|0.0000|MRPC|
|0.6275|0.0172|SST-2|0.8440|0.0000|
|MNLI|0.2731|0.0000|QNLI|0.3953|
|0.0000|AGNEWS|0.7185|0.0000|SQUAD|
|0.2104|0.0057||||
functional networks. The results indicate that the masking
of neurons within these functional networks leads to varying
degrees of performance degradation. Remarkably, masking
just a few dozen critical neurons can significantly impair the
model's performance. In some instances, masking as few as
one-thousandth of the total neurons is enough to cause a substantial drop in performance. In contrast, randomly masking
neurons has a minimal impact on the model's performance
as shown in Table 5. Our experiments demonstrate that even
when up to 15% neurons are randomly masked, performance
degradation remains insignificant, with the model's performance staying nearly identical to its baseline in most cases.
This finding is consistent with previous studies, which have
also observed that random neuron masking does not substantially affect model performance (AlKhamissi et al., 2024;
Song et al., 2024).
Table 6 and Table 7 present the performance results of
Qwen2.5-3B-Instruct and ChatGLM3-6B-base by masking
neurons within specific functional networks. These models
rely on well-designed prompts to generate appropriate text
for various tasks. When critical functional networks are
masked, the models' ability to generate coherent and taskrelevant text is severely compromised. For more details,
please refer to the supplementary material.
**Table 7. Performance of ChatGLM3-6B-base with Masked Functional Networks.**
|DATASETS|CHATGLM|NORMAL|CHATGLM|MASKED|
|---|---|---|---|---|
|10|COLA|0.6893|0.0000|MRPC|
|0.8161|0.0441|SST-2|0.9392|0.1422|
|MNLI|0.2062|0.0000|QNLI|0.1067|
|0.0053|AGNEWS|0.9128|0.0025|SQUAD|
|0.9021|0.0358||||
**Table 8. Performance of GPT-2 with Preserved Functional Networks. The first row of the table indicates the number of functional**
networks obtained from ICA decomposition, which corresponds
to the second to fifth columns. The first row for each dataset corresponds to the evaluation metric, while the second row indicates
the number of neurons obtained.
10
64
128
256
512
CoLA
0.3078
0.3509
0.4976
0.7776
0.6088
145
495
1221
2109
1409
MRPC
0.6838
0.6985
0.8113
0.7279
0.8137
71
394
759
426
817
QQP
0.6106
0.6849
0.8477
0.8766
0.8048
60
300
582
865
632
SST-2
0.5195
0.7970
0.6823
0.8394
0.9163
58
391
253
560
1357
MNLI
0.3220
0.3407
0.4158
0.6858
0.5294
61
342
722
1242
816
QNLI
0.5338
0.8559
0.8737
0.8629
0.8720
213
2111
3168
2568
3651
AGNEWS
0.6849
0.7039
0.8462
0.8596
0.8613
86
394
830
1281
1606
SQuAD
0.0016
0.2143
0.3314
0.3391
0.3398
79
652
945
1262
1577
### 4.3. Neuron Preservation Experiment
In addition to verifying the importance of these neurons by
masking functional networks, we also conducted a preservation experiment to further validate the significance of these
functional networks. In this section, we incrementally increase the number of functional networks obtained through
ICA decomposition, taking the union set of neurons within
these functional networks to increase the number of retained
neurons. We then mask the remaining less important neurons and observe how the model's performance changes.
Table 8, Table 9 and Table 10 show the performance changes
of the GPT-2, Qwen2.5-3B-Instruct, and ChatGLM3-6Bbase models as the number of decomposed functional networks increases. With the addition of more decomposed
functional networks, the models exhibit a gradual improvement in performance, moving from lower to higher performance metrics, which is also shown in Figure 3. Across

<!-- page:8 -->
**Table 9. Performance of Qwen2.5-3B-Instruct with Preserved**
|Functional|Networks.|
|---|---|
|10|64|
|128|256|
|512|COLA|
|0.0000|0.6913|
|0.6913|0.6913|
|0.6913|1500|
|6694|8330|
|10602|6349|
|MRPC|0.0000|
|0.6814|0.5245|
|0.6373|0.6324|
|1219|7556|
|10390|12042|
|12733|SST-2|
|0.0092|0.8200|
|0.8830|0.8888|
|0.8899|914|
|4592|7223|
|9686|11027|
|MNLI|0.0014|
|0.2546|0.2476|
|0.2740|0.2725|
|2116|8319|
|10064|11285|
|12080|QNLI|
|0.0002|0.2720|
|0.4077|0.3948|
|0.3967|750|
|7569|9478|
|10368|11043|
|AGNEWS|0.1201|
|0.0061|0.7280|
|0.7205|0.7213|
|1876|7423|
|10274|11491|
|12118|SQUAD|
|0.0057|0.1900|
|0.1918|0.2102|
|0.2106|2053|
|8436|10765|
|12272|13183|
**Table 10. Performance of ChatGLM3-6B-base with Preserved**
|Functional|Networks.|
|---|---|
|10|64|
|128|256|
|512|COLA|
|0.0000|0.6913|
|0.6913|0.6913|
|0.6913|2967|
|12306|15966|
|20495|27832|
|MRPC|0.0000|
|0.7770|0.8186|
|0.8162|0.8162|
|2344|13917|
|18520|23369|
|28697|SST-2|
|0.0000|0.8624|
|0.9300|0.9381|
|0.9404|1235|
|9390|14854|
|18702|26513|
|MNLI|0.0000|
|0.2123|0.2041|
|0.2062|0.2103|
|3399|12569|
|15586|20103|
|25080|QNLI|
|0.0000|0.1545|
|0.1157|0.1078|
|0.1067|905|
|13846|22341|
|28645|30838|
|AGNEWS|0.0000|
|0.9036|0.9087|
|0.9128|0.9128|
|2098|11831|
|18540|26489|
|32864|SQUAD|
|0.0001|0.8655|
|0.9016|0.9023|
|0.9021|2849|
|16150|21903|
|26646|30226|
all models, it is clear that the number of neurons identified as important increases as more functional networks are
incorporated. Our analysis reveals that, in practice, only
approximately 10% of the total neurons in the model are necessary to maintain performance at an acceptable level. This
observation is crucial because it indicates that there may
be substantial opportunities to improve the efficiency of AI
models. By identifying and utilizing only the most critical
neurons, we can potentially develop methods for creating
more resource-efficient models without sacrificing performance. This insight could pave the way for advancements
in AI model optimization, leading to lower computational
costs and greater accessibility of AI technologies.
## 5. Discussion and Conclusion
The present study introduces a novel approach to understanding LLM by applying methods inspired by the research
of cognitive neuroscience on FBN. Our findings reveal that
> **Figure 3.** The performance changes of the models as the number
of decomposed functional networks increases.
LLMs exhibit similar functional patterns as observed in human brains, which are crucial for their effective operation.
One of our key observations shows that only a small subset
of these functional networks (approximately 10% of the total
neurons) are necessary to maintain satisfactory performance
levels. By identifying and leveraging these critical networks,
future research could focus on developing more resourceefficient models with lower computational cost.
Our interdisciplinary approach, which combines insights
from cognitive neuroscience and artificial intelligence, offers a promising direction for future research. The application of neuroscience concepts to AI not only enhances
our understanding of LLM mechanisms but also opens new
avenues for innovation. For instance, techniques used to
identify FBNs in the human brain can be adapted to improve the interpretability and efficiency of AI models. By
drawing parallels between cognitive neuroscience and AI,
our research underscores the value of interdisciplinary collaboration. Future work should continue to explore how
neuroscience principles can inform AI design and optimization. Such efforts hold the promise of advancing both fields,
leading to more powerful and efficient AI systems that can
better serve diverse applications.
Our study also has some limitations. The algorithm employed in this research is ICA, which has various derivatives
such as canICA (Varoquaux et al., 2010b). These variants
can improve model's performance based on the unique characteristics of the dataset. Consequently, there is potential for
the development of novel algorithms tailored specifically to
the properties of LLMs.
Last but not least, our investigation focused solely on the
MLP layers. Future work could extend this approach to
other components within these models, such as attention
mechanisms or embedding layers. By broadening the scope
of analysis to include these additional modules, we can gain
a more comprehensive understanding of how functional
networks manifest in different components of LLMs.
References
AlKhamissi, B., Tuckute, G., Bosselut, A., and Schrimpf, M.
The llm language network: A neuroscientific approach

<!-- page:9 -->
for identifying causally task-relevant units. arXiv preprint
Amini, A., Gabriel, S., Lin, P., Koncel-Kedziorski, R., Choi,
Y., and Hajishirzi, H. Mathqa: Towards interpretable math
word problem solving with operation-based formalisms.
arXiv preprint arXiv:1905.13319, 2019.
Beckmann, C. F., DeLuca, M., Devlin, J. T., and Smith,
S. M. Investigations into resting-state connectivity using
independent component analysis. Philosophical Transactions of the Royal Society B: Biological Sciences, 360
(1457):1001-1013, 2005.
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,
Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., et al. Language models are few-shot learners.
Advances in neural information processing systems, 33:
1877-1901, 2020.
Bullmore, E. and Sporns, O. Complex brain networks: graph
theoretical analysis of structural and functional systems.
Nature reviews neuroscience, 10(3):186-198, 2009.
Chen, Y., Cao, P., Chen, Y., Liu, K., and Zhao, J. Journey
to the center of the knowledge neurons: Discoveries of
language-independent knowledge neurons and degenerate
knowledge neurons. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 17817-17825, 2024.
Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., and
Wei, F.
Knowledge neurons in pretrained transformers. In Muresan, S., Nakov, P., and Villavicencio, A.
(eds.), Proceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pp. 8493-8502, Dublin, Ireland, May
## 2022. Association for Computational Linguistics. doi:
10.18653/v1/2022.acl-long.581.
Dong, Q., Qiang, N., Lv, J., Li, X., Liu, T., and Li, Q.
Discovering functional brain networks with 3d residual
autoencoder (resae). In Medical Image Computing and
Computer Assisted Intervention-MICCAI 2020: 23rd International Conference, Lima, Peru, October 4-8, 2020,
Proceedings, Part VII 23, pp. 498-507. Springer, 2020.
Elfwing, S., Uchibe, E., and Doya, K. Sigmoid-weighted
linear units for neural network function approximation
in reinforcement learning. Neural networks, 107:3-11,
2018.
Ge, B., Wang, H., Wang, P., Tian, Y., Zhang, X., and Liu, T.
Discovering and characterizing dynamic functional brain
networks in task fmri. Brain Imaging and Behavior, 14:
1660-1673, 2020.
GLM, T., Zeng, A., Xu, B., Wang, B., Zhang, C., Yin, D.,
Zhang, D., Rojas, D., Feng, G., Zhao, H., et al. Chatglm:
A family of large language models from glm-130b to
glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024.
Hassabis, D., Kumaran, D., Summerfield, C., and Botvinick,
M. Neuroscience-inspired artificial intelligence. Neuron,
95(2):245-258, 2017.
He, M., Hou, X., Ge, E., Wang, Z., Kang, Z., Qiang, N.,
Zhang, X., and Ge, B. Multi-head attention-based masked
sequence model for mapping functional brain networks.
Frontiers in Neuroscience, 17:1183145, 2023.
Hendrycks, D. and Gimpel, K. Gaussian error linear units
(gelus). arXiv preprint arXiv:1606.08415, 2016.
Huo, J., Yan, Y., Hu, B., Yue, Y., and Hu, X. MMNeuron:
Discovering neuron-level domain-specific interpretation
in multimodal large language model. In Al-Onaizan,
Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of
the 2024 Conference on Empirical Methods in Natural
Language Processing, pp. 6801-6816, Miami, Florida,
USA, November 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.emnlp-main.387.
Hyv¨arinen, A. and Oja, E. Independent component analysis:
algorithms and applications. Neural networks, 13(4-5):
411-430, 2000.
Liu, Y., Ge, E., Qiang, N., Liu, T., and Ge, B. Spatialtemporal convolutional attention for mapping functional
brain networks. In 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), pp. 1-4. IEEE,
2023a.
Liu, Y., Han, T., Ma, S., Zhang, J., Yang, Y., Tian, J., He,
H., Li, A., He, M., Liu, Z., et al. Summary of chatgptrelated research and perspective towards the future of
large language models. Meta-Radiology, pp. 100017,
2023b.
Liu, Y., Ge, E., He, M., Liu, Z., Zhao, S., Hu, X., Qiang,
N., Zhu, D., Liu, T., and Ge, B. Mapping dynamic spatial patterns of brain function with spatial-wise attention.
Journal of Neural Engineering, 21(2):026005, 2024a.
Liu, Y., Ge, E., Kang, Z., Qiang, N., Liu, T., and Ge, B.
Spatial-temporal convolutional attention for discovering
and characterizing functional brain networks in task fmri.
NeuroImage, 287:120519, 2024b.
Liu, Y., He, H., Han, T., Zhang, X., Liu, M., Tian, J., Zhang,
Y., Wang, J., Gao, X., Zhong, T., Pan, Y., Xu, S., Wu,
Z., Liu, Z., Zhang, X., Zhang, S., Hu, X., Zhang, T.,
Qiang, N., Liu, T., and Ge, B. Understanding llms: A
comprehensive overview from training to inference. Neurocomputing, 620:129190, 2025.

<!-- page:10 -->
Logothetis, N. K. What we can do and what we cannot do
with fmri. Nature, 453(7197):869-878, 2008.
Lundstrom, D. D., Huang, T., and Razaviyayn, M. A rigorous study of integrated gradients method and extensions
to internal neuron attributions. In International Conference on Machine Learning, pp. 14485-14508. PMLR,
2022.
Lv, J., Jiang, X., Li, X., Zhu, D., Chen, H., Zhang, T., Zhang,
S., Hu, X., Han, J., Huang, H., et al. Sparse representation
of whole-brain fmri signals for identification of functional
networks. Medical image analysis, 20(1):112-134, 2015.
Matthews, P. M. and Jezzard, P. Functional magnetic resonance imaging. Journal of Neurology, Neurosurgery &
Psychiatry, 75(1):6-12, 2004.
Mensch, A., Varoquaux, G., and Thirion, B. Compressed online dictionary learning for fast resting-state fmri decomposition. In 2016 IEEE 13th International Symposium on
Biomedical Imaging (ISBI), pp. 1282-1285. IEEE, 2016.
Merity, S., Xiong, C., Bradbury, J., and Socher, R.
Pointer sentinel mixture models.
arXiv preprint
Miller, E. K. and Cohen, J. D. An integrative theory of
prefrontal cortex function. Annual review of neuroscience,
24(1):167-202, 2001.
Mountcastle, V. B. The columnar organization of the neocortex. Brain: a journal of neurology, 120(4):701-722,
1997.
Nikankin, Y., Reusch, A., Mueller, A., and Belinkov,
Y. Arithmetic without algorithms: Language models
solve math with a bag of heuristics.
arXiv preprint
Niu, J., Liu, A., Zhu, Z., and Penn, G. What does the
knowledge neuron thesis have to do with knowledge?
In The Twelfth International Conference on Learning
Representations, 2024.
Puri, R., Kung, D. S., Janssen, G., Zhang, W., Domeniconi,
G., Zolotov, V., Dolby, J., Chen, J., Choudhury, M. R.,
Decker, L., Thost, V., Buratti, L., Pujar, S., Ramji, S.,
Finkler, U., Malaika, S., and Reiss, F. Codenet: A largescale AI for code dataset for learning a diversity of coding
tasks. In NeurIPS Datasets and Benchmarks, 2021.
Radford, A., Narasimhan, K., Salimans, T., Sutskever, I.,
et al. Improving language understanding by generative
pre-training. OpenAI, 2018.
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D.,
Sutskever, I., et al. Language models are unsupervised
multitask learners. OpenAI blog, 1(8):9, 2019.
Raichle, M. E. The brain's default mode network. Annual
review of neuroscience, 38(1):433-447, 2015.
Raichle, M. E. and Snyder, A. Z. A default mode of brain
function: a brief history of an evolving idea. Neuroimage,
37(4):1083-1090, 2007.
Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. SQuAD:
100,000+ questions for machine comprehension of text.
In Su, J., Duh, K., and Carreras, X. (eds.), Proceedings
of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383-2392, Austin, Texas,
November 2016. Association for Computational Linguistics.
Smith, S. M., Fox, P. T., Miller, K. L., Glahn, D. C., Fox,
P. M., Mackay, C. E., Filippini, N., Watkins, K. E., Toro,
R., Laird, A. R., et al. Correspondence of the brain's
functional architecture during activation and rest. Proceedings of the national academy of sciences, 106(31):
13040-13045, 2009.
Song, R., He, S., Jiang, S., Xian, Y., Gao, S., Liu, K., and
Yu, Z. Does large language model contain task-specific
neurons?
In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing, pp.
7101-7113, 2024.
Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribution for deep networks. In International conference on
machine learning, pp. 3319-3328. PMLR, 2017.
Tang, T., Luo, W., Huang, H., Zhang, D., Wang, X., Zhao,
X., Wei, F., and Wen, J.-R. Language-specific neurons:
The key to multilingual capabilities in large language
models. arXiv preprint arXiv:2402.16438, 2024.
Varoquaux, G., Keller, M., Poline, J.-B., Ciuciu, P., and
Thirion, B. Ica-based sparse features recovery from fmri
datasets.
In 2010 IEEE International Symposium on
Biomedical Imaging: From Nano to Macro, pp. 1177-
## 1180. IEEE, 2010a.
Varoquaux, G., Sadaghiani, S., Pinel, P., Kleinschmidt, A.,
Poline, J.-B., and Thirion, B. A group model for stable
multi-subject ica on fmri datasets. Neuroimage, 51(1):
288-299, 2010b.
Vaswani, A. Attention is all you need. Advances in Neural
Information Processing Systems, 2017.
Vilas, M. G., Adolfi, F., Poeppel, D., and Roig, G. Position: An inner interpretability framework for ai inspired
by lessons from cognitive neuroscience. In Forty-first
International Conference on Machine Learning.
Wang, A. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint

<!-- page:11 -->
Wang, J., Shi, E., Hu, H., Ma, C., Liu, Y., Wang, X., Yao,
Y., Liu, X., Ge, B., and Zhang, S. Large language models
for robotics: Opportunities, challenges, and perspectives.
Journal of Automation and Intelligence, 2024.
Wang, X., Wen, K., Zhang, Z., Hou, L., Liu, Z., and Li,
J. Finding skill neurons in pre-trained transformer-based
language models. In Proceedings of EMNLP, 2022.
Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C.,
Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin,
H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J.,
Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K.,
Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P.,
Wang, P., Peng, R., Men, R., Gao, R., Lin, R., Wang, S.,
Bai, S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng,
X., Zhou, X., Ren, X., Zhang, X., Wei, X., Ren, X., Fan,
Y., Yao, Y., Zhang, Y., Wan, Y., Chu, Y., Liu, Y., Cui, Z.,
Zhang, Z., and Fan, Z. Qwen2 technical report. arXiv
preprint arXiv:2407.10671, 2024a.
Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li,
C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical
report. arXiv preprint arXiv:2412.15115, 2024b.
Yu, Z. and Ananiadou, S. Interpreting arithmetic mechanism
in large language models through comparative neuron
analysis. arXiv preprint arXiv:2409.14144, 2024a.
Yu, Z. and Ananiadou, S. Neuron-level knowledge attribution in large language models. In Proceedings of the 2024
Conference on Empirical Methods in Natural Language
Processing, pp. 3267-3280, 2024b.
Zhang, X., Zhao, J., and LeCun, Y. Character-level convolutional networks for text classification. Advances in neural
information processing systems, 28, 2015.
Zhao, H., Chen, H., Yang, F., Liu, N., Deng, H., Cai, H.,
Wang, S., Yin, D., and Du, M. Explainability for large
language models: A survey. ACM Transactions on Intelligent Systems and Technology, 15(2):1-38, 2024a.
Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y.,
Min, Y., Zhang, B., Zhang, J., Dong, Z., et al. A survey of
large language models. arXiv preprint arXiv:2303.18223,
2023.
Zhao, Y., Zhang, W., Chen, G., Kawaguchi, K., and Bing,
L. How do large language models handle multilingualism? In The Thirty-eighth Annual Conference on Neural
Information Processing Systems, 2024b.

<!-- page:12 -->
A. Appendix
A.1. Neuron Lesion Experiment
Figures 4 and 5 present the neuron lesion study conducted on the ChatGLM3-6B-base model for the task of long-text
summarization. Specifically, the study involves randomly masking neurons within the model and comparing the performance
with masked neurons that are part of functional networks.
Prompt:
Summarize the following article in brief:(CNN)The Palestinian Authority officially became the 123rd member of the
International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian
territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based.
The Palestinians signed the ICC\'s founding Rome Statute in January, when they also accepted its jurisdiction over alleged
crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014." Later that month,
the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war
crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well.
Israel and the United States, neither of which is an ICC member, opposed the Palestinians\' efforts to join the body. But
Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday\'s ceremony, said it was a move toward greater justice.
"As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long
era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared
goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the
first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine
acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are
substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights Watch welcomed the
development. "Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and
countries that support universal acceptance of the court\'s treaty should speak out to welcome its membership," said
Balkees Jarrah, international justice counsel for the group. "What\'s objectionable is the attempts to undermine
international justice, not Palestine\'s decision to join a treaty to which over 100 countries around the world are
members." In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described
it as an outrage, saying the court was overstepping its boundaries. The United States also said it "strongly" disagreed
with the court\'s decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do
not believe that it is eligible to join the ICC," the State Department said in a statement. It urged the warring sides to
resolve their differences through direct negotiations. "We will continue to oppose actions against Israel at the ICC as
counterproductive to the cause of peace," it said. But the ICC begs to differ with the definition of a state for its
purposes and refers to the territories as "Palestine." While a preliminary examination is not a formal investigation, it
allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda
said her office would "conduct its analysis in full independence and impartiality." The war between Israel and Hamas
militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since
June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes.
CNN\'s Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.
The Palestinian Authority has officially become the 123rd member of the International Criminal Court. This gives the court
jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague,
in the Netherlands, where the court is based. The Palestinians signed the ICC's founding Rome Statute in January and
accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem,
since June 13, 2014." The ICC opened a preliminary examination into the situation in Palestinian territories, paving the
way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to
counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts
to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a move
toward greater justice.
Masked 15% ChatGLM3-6B Neurons:
Reference Answer:
Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June .\nIsrael
and the United States opposed the move, which could open the door to war crimes investigations against Israelis.
> **Figure 4.** Random masked 15% neurons in ChatGLM3-6B-base.

<!-- page:13 -->
Components 1: activate neuron num :4016/114688
The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, as the court
gained jurisdiction over alleged crimes in Palestinian territories. Palestinians may be subject to counter-charges as well
as possible war crimes investigations against Israelis. Israel and the United States opposed the move, but Palestinian
Foreign Minister Riad al-Malki said it was a move toward greater justice. Palestine acquied all the rights as well as
responsiveness as a State Party to the ICC, as the preliminary examination was opened in January.
Components 2: activate neuron num :2791/114688
16 16 (2016)
Components 3: activate neuron num :2141/114688
I apologize for the错ful in the OPEN thread.
Components 4: activate neuron num :1906/114688
The Palestinian Authority has become the 123nd member of the International Criminal Crime 是非常非常重要的 the ICC. The
Palestinian簽ed the ICC's founding Rome Statuate in January, which also marked as theICA's jurisdiction over Allegations of
acne in Palestinian charge. PalestinianForecommin Foreign Minister Riad al-Malki said that it was a step Towards greater
justice. TheICCA and Israel and the United States, shun 我们就不得不考虑它出现了的事实. TheICCAICICICed the Palestinian's
efforts to join the group. .
Components 5: activate neuron num :2925/114688
The 11,009,000 12,000,000 and 12,000,000 12,000,000 and 12,000,000 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000
and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and
12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and
12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and
12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and 12,000,000 and
12,000,000 and 12,000,000 and
Components 6: activate neuron num :2663/114688
The formal Jewish paper 0 2002-1-25 0沙敢: this is part of the plan of the事, or catch the 19-year-statement re: peace,
please安全保障 1000 love on;m摘要是一 brief:the sand is the equivalent size.
Components 7: activate neuron num :3601/114688
'.
Components 8: activate neuron num :2590/114688
The international
Components 9: activate neuron num :1795/114688
渡'));" next to
Components 10: activate neuron num :761/114688
The 123rd member of the International Criminal Court (ICCC) is the Palestinian Authority, which has accepted the ICC's
founding Rome Statute in January, and is now a State Party to the treaty. The Palestinians may be subject to countercharges as well. The ICC opened a preliminary examination into the situation in Palestinian territories, and the process is
expected to lead to possible war crimes investigations against Israelis. The Israeli and US opposition to the Palestinians'
efforts to join the body has been脂肪, and the process is expected to lead to war. The process is expected to lead to
possible war crimes investigations against Israelis. The process is expected to lead to possible war crimes investigations
against Israelis.
Masked 10 Functional Networks Respectively:
> **Figure 5.** Masked 10 functional networks respectively in ChatGLM3-6B-base.
