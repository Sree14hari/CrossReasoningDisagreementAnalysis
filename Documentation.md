# Project Documentation: Cross-Reasoning Disagreement Analysis (CRDA)

## 1. Problem Formulation and Research Vision

**Project Title:** CRDA: Cross-Reasoning Disagreement Analysis for Uncertainty Quantification in Hybrid Deep Neural Networks
**Research Domains:** Explainable AI (XAI), Uncertainty Quantification (UQ), Hybrid Deep Learning, Trustworthy AI

### 1.1 The Vulnerability in Current Uncertainty Quantification

Deep neural networks deployed in high-stakes environments necessitate rigorous Uncertainty Quantification (UQ). Traditional approaches—including Deep Ensembles, Monte Carlo (MC) Dropout, and Temperature Scaling—derive uncertainty metrics exclusively from the output prediction layer.

Modern hybrid architectures fuse heterogeneous reasoning mechanisms to improve accuracy. However, this introduces an unmonitored vulnerability: **internal representation divergence**. Distinct reasoning modules may inherently disagree on feature importance, yet the final unified classifier can still produce an overconfident, incorrect prediction. Output-layer UQ methods are blind to this internal representational conflict.

### 1.2 Core Research Hypotheses

* **H1 (The Disagreement Hypothesis):** Prediction reliability in hybrid neural networks is inversely correlated with the representational disagreement between their heterogeneous reasoning modules.
* **H2 (The Early Detection Hypothesis):** Internal reasoning disagreement provides a stronger signal for identifying out-of-distribution (OOD) samples and model failures than output-layer entropy alone.
* **H3 (The Efficiency Hypothesis):** CRDA aims to provide a favorable trade-off between uncertainty estimation quality and computational efficiency.

### 1.3 Notation Table

| Symbol | Definition |
| --- | --- |
| $x$ | Model input (e.g., image, audio, sequence) |
| $J$ | Total number of heterogeneous reasoning modules |
| $g_i$ | Feature extraction function for reasoning module $i$ |
| $X_i$ | Intermediate feature representation from module $i$ |
| $\Psi_i$ | Descriptor extraction function for module $i$ |
| $D_i$ | Extracted descriptor for module $i$ |
| $\Phi_i$ | Projection mapping function into the latent space |
| $K$ | Dimensionality of the Shared Latent Reasoning Space |
| $\tau$ | Uncertainty rejection threshold |
| $U_{CRDA}$ | Final calculated internal disagreement score |

---

## 2. Theoretical Framework and Mathematical Formulation

To construct a generalized, architecture-agnostic methodology, CRDA isolates the mathematical flow into distinct stages: Topology Definition, Descriptor Extraction, Projection, Normalization, and Divergence Calculation.

### 2.1 Supported Hybrid Topologies and Semantic Alignment

CRDA is designed to monitor disagreement across varied architectural configurations. To ensure valid representational comparisons, intermediate features $X_i$ must be extracted at **comparable semantic depths** (e.g., extracting features post-Block-4 of a ResNet branch alongside post-Block-4 of a ViT branch). The framework supports three core hybrid topologies:

1. **Parallel Topologies:** Independent reasoning branches (e.g., a CNN and a Transformer) that process the input simultaneously before late fusion.
2. **Sequential Topologies:** Architectures where a localized module feeds into a global module (e.g., CNN into LSTM). CRDA extracts features post-local-processing and post-global-processing prior to the final fully connected layers.
3. **Cross-Attention Fusion Topologies:** Models that actively share tokens between modules. CRDA extracts features immediately before the cross-attention bottlenecks to capture maximum branch independence.

### 2.2 Input and Output Specifications

* **Inputs:** Let $\mathcal{M}$ be a hybrid network with $J$ reasoning modules. For a given input $x$, intermediate representations are extracted as $X_i = g_i(x)$, matching the semantic depth constraints.
* **Outputs:** The framework outputs three distinct elements: the class probabilities $\mathbf{y} \in \mathbb{R}^C$, the final predicted label $\hat{y}$, and the CRDA uncertainty score $U_{CRDA} \in [0,1]$.

### 2.3 Dual-Mode Descriptor Extraction

To unify varying architectural topologies, we define a generic descriptor function $D_i = \Psi_i(X_i, [\nabla X_i])$. To balance performance and overhead, CRDA defines two operational modes:

* **CRDA-Fast:** Uses activation-only descriptors. Guarantees single-forward-pass efficiency ($O(1)$ overhead).
* **CRDA-Precise:** Uses gradient-weighted descriptors ($\nabla X_i$) for maximum fidelity, requiring a targeted backward pass (vector-Jacobian product).

**1. Convolutional Neural Networks (CNNs):**
For a spatial feature map $X_{CNN} \in \mathbb{R}^{H \times W \times C}$, the specific implementation of $\Psi$ computes spatial feature energy across channels (Fast mode):


$$D_{CNN}^{(c)} = \Psi_{CNN}(X_{CNN})^{(c)} = \frac{1}{H \times W} \sum_{h=1}^{H} \sum_{w=1}^{W} (X_{CNN, h, w}^{(c)})^2$$

**2. Transformers:**
Relying on raw attention scores can be misleading. Depending on the mode, CRDA evaluates Transformers as follows:

* *CRDA-Fast:* Uses Attention Rollout or a fixed linear probe on the CLS token to faithfully proxy feature importance without backpropagation.
* *CRDA-Precise:* Extracts gradient-weighted class activation (akin to Grad-CAM for Transformers), utilizing the gradient of the predicted class $y^c$ with respect to the attention maps $A$:

$$D_{Trans} = \Psi_{Trans}(X_{Trans}, \nabla X_{Trans}) = \text{ReLU} \left( \sum_k \alpha_k A^k \right)$$



where $\alpha_k$ are the gradients pooling weights.

**3. Sequential and State-Space Models:**

* **LSTM:** $\Psi_{LSTM}(X_{LSTM})$ computes the temporal variance across hidden states.
* **Mamba:** $\Psi_{Mamba}(X_{Mamba})$ extracts the state-space transition weights parameterized over the sequence length.

### 2.4 Subspace Projection

Raw descriptors $D_i$ exist in structurally divergent dimensionalities (e.g., $C$ channels vs. $N$ tokens). We introduce mapping functions $\Phi_i$ to project these descriptors into a Shared Latent Reasoning Space $\mathcal{S} \subseteq \mathbb{R}^K$.

$$Z_i = \Phi_i(D_i)$$

**Theoretical Justification for Adaptive Average Pooling:**
The default projection $\Phi_i$ utilizes Adaptive Average Pooling. This non-parametric approach acts as a piecewise smoothing function, compressing or expanding the vector into exactly $K$ bins while strictly preserving the relative magnitude and density distribution of the original descriptor.

### 2.5 Normalization

The projected features $Z_i \in \mathbb{R}^K$ are converted into valid probability distributions over the shared space using Softmax normalization: $\mathbf{\hat{p}}_i = \text{Softmax}(Z_i)$. This ensures differentiability and prevents negative values from distorting the divergence computation, unlike strict $L_1$-normalization.

### 2.6 Generalized Divergence Computation

To future-proof the methodology for architectures with $J \ge 2$ modules, the internal disagreement score $U_{CRDA}$ utilizes the **Generalized Jensen-Shannon Divergence ($D_{JS}$)**. Let $\mathbf{M} = \frac{1}{J} \sum_{i=1}^{J} \mathbf{\hat{p}}_i$ be the mean distribution across all projected reasoning branches:


$$U_{CRDA} = D_{JS}(\mathbf{\hat{p}}_1, \dots, \mathbf{\hat{p}}_J) = \frac{1}{J} \sum_{i=1}^{J} D_{KL}(\mathbf{\hat{p}}_i \parallel \mathbf{M})$$

### 2.7 Theoretical Guarantees

**Proposition 1 (Properties of $U_{CRDA}$):**

1. **Identity / Definiteness:** $U_{CRDA} \ge 0$, with $U_{CRDA} = 0$ if and only if the projected reasoning distributions are perfectly identical.
2. **Symmetry:** The metric is invariant to the permutation of the $J$ reasoning modules.
3. **Boundedness:** The maximum possible divergence between any $\mathbf{\hat{p}}_i$ and the mean $\mathbf{M}$ across $J$ distributions is $\log(J)$. By computing the Kullback-Leibler divergence using a base-$J$ logarithm (or equivalently, dividing the natural-log result by $\log(J)$), the score is strictly normalized such that $U_{CRDA} \in [0, 1]$.

---

## 3. Threshold Calibration

The deployment of CRDA requires the calibration of the rejection threshold $\tau$ on a held-out validation set post-training. Methods include ECE Optimization, Percentile Rejection, and Risk-Tolerance Tuning based on domain-specific cost margins.

---

## 4. Algorithmic Implementation

```text
Algorithm 1: Cross-Reasoning Disagreement Analysis (CRDA) Inference
-------------------------------------------------------------------
Input: Input data x, Hybrid model M with J modules, Extractor functions g_i, 
       Descriptor functions \Psi_i, Projections \Phi_i, Threshold \tau, Mode (Fast/Precise)
Output: Class probabilities y, Predicted label \hat{y}, Uncertainty U_{CRDA}

1: y, \hat{y} = M.forward(x)
2: if Mode == Precise then
3:     Compute gradients \nabla X_i for relevant modules
4: end if
5: for i = 1 to J do
6:     X_i = g_i(x)                             // Extract representation at aligned depth
7:     D_i = \Psi_i(X_i, [\nabla X_i])          // Compute reasoning descriptor
8:     Z_i = \Phi_i(D_i)                        // Project to dimension K
9:     \hat{p}_i = Softmax(Z_i)                 // Normalize
10: end for
11: U_{CRDA} = D_{JS}(\hat{p}_1, \dots, \hat{p}_J)   // Compute generalized divergence

12: if U_{CRDA} > \tau then
13:    Flag prediction as UNRELIABLE
14: else
15:    Flag prediction as RELIABLE
    
16: return y, \hat{y}, U_{CRDA}

```

---

## 5. Experimental Design and Baselines

### 5.1 Strengthened Uncertainty Baselines

CRDA will be benchmarked against both output-level and representation-level uncertainty methodologies:

* **Output-Level UQ:** Deep Ensembles, Monte Carlo (MC) Dropout, Temperature Scaling.
* **Representation-Level UQ:** Mahalanobis Distance-based Confidence (applied to the penultimate layer) and Representation Jensen-Shannon Divergence.

### 5.2 OOD and Robustness Benchmarks

To prove CRDA's viability as an early-detection failure mechanism, models will be evaluated across precise shifts:

* **Vision (Corruptions):** ImageNet-C (testing resilience against noise, blur, and weather).
* **Vision (Out-of-Distribution):** ImageNet-O, with MNIST/CIFAR serving as "far-OOD" boundary tests.
* **Audio:** Environmental Sound Classification (ESC-50) evaluated with injected Gaussian noise and cross-lingual vocal shifts.

---

## 6. Evaluation and Qualitative Analysis

### 6.1 Calibration Metrics

* **Metrics:** Expected Calibration Error (ECE), Brier Score, AUROC for OOD detection, FPR95.
* **Statistical Significance:** Paired Wilcoxon signed-rank testing will ensure the performance gains against baselines are statistically rigorous.

### 6.2 Confidence-Disagreement Quadrant Analysis

To effectively illustrate CRDA's value proposition, we introduce a $2 \times 2$ Confidence-Disagreement Matrix comparing the Softmax Output Confidence against the Internal $U_{CRDA}$ Disagreement Score:

|  | **Low Disagreement (Low $U_{CRDA}$)** | **High Disagreement (High $U_{CRDA}$)** |
| --- | --- | --- |
| **High Softmax Confidence** | **Quadrant I: True Reliables**  The model is confident, modules agree, and the prediction is highly accurate. | **Quadrant II: Overconfident Hallucinations**  The critical failure mode. Softmax is erroneously high, but internal conflict correctly flags the prediction as unsafe. |
| **Low Softmax Confidence** | **Quadrant III: Ambiguous Data**  Hard edge-cases where reasoning paths agree on the difficulty, resulting in a naturally low-confidence output. | **Quadrant IV: Known Unknowns (OOD)**  Ideal OOD detection. The model lacks confidence, and modules severely diverge on feature interpretation. |

Highlighting **Quadrant II** will be the focal point of the qualitative results, demonstrating how CRDA catches fatal, high-confidence errors that traditional UQ methods blindly accept.

### 6.3 Projection Ablations

Evaluating Adaptive Average Pooling (Default) against Principal Component Analysis (PCA) and Random Projection matrices for mapping features into the Shared Latent Reasoning Space.

---

## 7. Limitations and Future Work

### 7.1 Framework Limitations

* **Architectural Dependencies:** CRDA fundamentally requires transparent access to intermediate feature representations, limiting utility in closed-source API environments.
* **Memory Overhead:** Caching and backpropagating (for CRDA-Precise) during the forward pass incurs memory overhead proportional to tensor sizes and $K$.

### 7.2 Future Scope

* **Learnable Alignment:** Investigating unsupervised manifold alignment techniques to dynamically determine optimal projection parameters.
* **Graph-Based Extensions:** Adapting descriptor extraction functions ($\Psi_i$) for Graph Neural Networks (GNNs) to evaluate structural reasoning disagreement.