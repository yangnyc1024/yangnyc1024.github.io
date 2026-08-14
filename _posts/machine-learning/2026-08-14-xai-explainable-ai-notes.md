---
title: "xAI (Explainable AI)"
date: 2026-08-14 15:00:00 +0800
categories:
  - Explainable AI
  - SHAP
  - LIME
writing_category: machine-learning
tags: [xai, explainable-ai, shap, lime, interpretability, ebm]
description: "Notes on explainable AI: SHAP vs. LIME, why Shapley values are relative to a baseline and expensive to compute exactly, LEAF's framework for scoring explanation quality, global surrogate models, PDP, and the Explainable Boosting Machine."
---

Explainable AI (xAI) refers to methods and techniques that make the behavior and predictions of machine learning models understandable to humans. As models get more complex (deep nets, gradient-boosted ensembles, etc.), the "black box" problem grows — we can see the output but not *why* the model produced it. xAI methods try to open that box, either by explaining a **single prediction** (local explanation) or the **model's overall behavior** (global explanation).

*可解释 AI（xAI）研究如何让模型的预测变得"人能看懂"，分为解释单条预测的**局部解释**和解释模型整体行为的**全局解释**两大类。*

---

## 1. What is SHAP? What is LIME? How are they different?

**LIME (Local Interpretable Model-agnostic Explanations，局部可解释模型无关解释)** explains a single prediction by perturbing the input, getting the black-box model's predictions on those perturbed samples, weighting them by proximity to the original instance, and fitting a simple interpretable model (usually linear) on that weighted local neighborhood. The coefficients of that local linear model become the "explanation":

$$
\xi(x) = \underset{g \in G}{\arg\min} \; \mathcal{L}(f, g, \pi_x) + \Omega(g)
$$

$f$ is the black-box model, $g \in G$ a candidate interpretable model, $\pi_x(z)$ a locality weight (how close perturbed sample $z$ is to $x$), $\mathcal L$ the weighted fitting loss, and $\Omega(g)$ a complexity penalty on $g$ [1].

*LIME 在样本 $x$ 附近随机扰动生成一批点，用一个简单模型 $g$ 在这个局部邻域内拟合黑箱 $f$，$g$ 的系数就是解释；$\Omega(g)$ 用来约束 $g$ 别太复杂。*

**SHAP (SHapley Additive exPlanations，Shapley 可加解释)** is grounded in cooperative game theory: each feature is a "player" and SHAP computes each feature's **Shapley value** — its fair, averaged marginal contribution across all possible feature coalitions. SHAP frames the explanation as an **additive feature attribution model** [1]:

$$
g(z') = \phi_0 + \sum_{j=1}^{M} \phi_j z_j'
$$

where $z' \in \{0,1\}^M$ marks which features are "present," $\phi_j$ is feature $j$'s Shapley value, and $\phi_0 = E[\hat f(X)]$ is the baseline (average prediction). This gives the **local accuracy (efficiency)** identity:

$$
\hat f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j
$$

and **missingness**: $x_j' = 0 \Rightarrow \phi_j = 0$.

*SHAP 把解释写成"基准值 + 各特征贡献之和"的可加模型，$\phi_0$ 是背景平均预测，$\phi_j$ 是每个特征的 Shapley 值；local accuracy 说的就是所有贡献加起来正好等于最终预测。*

The general (game-theoretic) Shapley value formula, for $p$ features and a value function $val(S)$ over feature subset $S$ [2]:

$$
\phi_j(val)=\sum_{S\subseteq\{1,\ldots,p\} \setminus \{j\}}\frac{|S|!\left(p-|S|-1\right)!}{p!}\Big(val\left(S\cup\{j\}\right)-val(S)\Big)
$$

In the ML setting, $val_x(S)$ integrates out the features *not* in $S$ over their marginal distribution:

$$
val_{\mathbf{x}}(S)=\int\hat{f}(x_{1},\ldots,x_{p})\, d\mathbb{P}_{X_C}-\mathbb{E}[\hat{f}(\mathbf{X})]
$$

and since exact computation is intractable, a Monte-Carlo estimator averages the prediction difference over randomly-constructed "with/without feature $j$" sample pairs [2]:

$$
\hat{\phi}_{j} = \frac{1}{M}\sum_{m=1}^M\Big(\hat{f}(\mathbf{x}^{(m)}_{+j}) - \hat{f}(\mathbf{x}^{(m)}_{-j})\Big)
$$

*$val(S)$ 衡量"只知道子集 $S$ 的特征时预测偏离平均水平多少"；Shapley 公式对特征 $j$ 在所有可能加入顺序下的边际贡献做加权平均。由于精确遍历所有子集不可行，实际用蒙特卡洛抽样近似（随机构造"有/无特征 $j$"的样本对，取预测差的平均）。*

SHAP values satisfy four axioms — **Efficiency**（效率性，即 local accuracy）、**Symmetry**（对称性）、**Dummy**（哑元性：无贡献特征归因为零）、**Additivity**（可加性）— and Shapley values are the *unique* allocation satisfying all four [2].

**Key differences between LIME and SHAP:**

- **Foundation**: LIME is a heuristic local surrogate; SHAP's attributions are the unique solution to the fairness axioms above.
- **Consistency**: SHAP guarantees additive, consistent attributions; LIME has no such guarantee.
- **Unification (Kernel SHAP)**: plugging a specific kernel into LIME's own loss recovers the Shapley values exactly [1]:

$$
\pi_x(z') = \frac{(M-1)}{\binom{M}{|z'|}\, |z'|\,(M-|z'|)}, \qquad L(\hat f, g, \pi_x) = \sum_{z' \in Z} \big[\hat f(h_x(z')) - g(z')\big]^2 \, \pi_x(z')
$$

  i.e. Kernel SHAP is a correctly-weighted special case of LIME's loss function.
- **Cost**: LIME just fits one local regression; exact SHAP is exponentially expensive (see "sublinear" below).
- **Scope**: LIME is purely local; SHAP values also aggregate into global feature importance ($I_j = \frac1n\sum_i \|\phi_j^{(i)}\|$) and dependence plots.

*一句话总结——LIME 靠局部采样拟合简单模型，没有唯一性保证、结果不稳定；SHAP 靠博弈论公理保证唯一性和可加性，但计算更贵。Kernel SHAP 本质是"权重调对了"的 LIME，这是两者被"统一"的关键。*

### 1.1 The waterfall plot

The chart starts at the base value $\phi_0$ (average prediction) and shows each feature pushing the prediction up (red) or down (blue) until it reaches the actual prediction for this instance.

![SHAP waterfall plot showing how each feature pushes a house price prediction up or down from the base value to the final prediction](/assets/posts/xai-explainable-ai-notes/waterfall_plot.png)

Base value **22532.81** → after every feature's push (e.g. `number of rooms = 5.878` → −3497.13, `% working class = 16.2` → +1856.89, …) → final prediction `f(x) = 21022.57`. A direct visualization of `prediction = base value + sum(SHAP values)`.

*(This chart was regenerated at high resolution with the actual `shap` Python library, `shap.plots.waterfall()`, using the same feature names/values and SHAP contributions transcribed from the original note — the numbers match exactly, `f(x) = 21022.566 ≈ 21022.57`. The example itself is originally from [Aidan Cooper's guide to interpreting SHAP analyses](https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/) [10], not from Molnar's book.)*

### 1.2 Why is LIME not deterministic?

LIME perturbs the instance **randomly** to build the neighborhood it fits. Without a fixed random seed, each `explain_instance()` call samples different points, producing a different weighted regression — and potentially a different set of "top features" — each run [3]. This is a well-known instability, worse when the model's local decision boundary is highly non-linear or few samples are used. **DLIME** was proposed to fix this by replacing random sampling with clustering-based neighborhood selection (hierarchical clustering + KNN), so the same instance always gets the same neighborhood [4].

*LIME 每次都要重新随机采样扰动点，没固定种子的话，两次运行选出的"重要特征"可能都不一样；DLIME 用聚类代替随机采样来解决这个问题。*

---

## 2. "Relativity": SHAP values are relative to a baseline, not absolute

*(This bullet in the original notes was ambiguous — interpreted here as SHAP's efficiency/local-accuracy axiom, the closest well-established xAI concept.)*

A SHAP value is never an absolute measure of a feature's importance in isolation — it's always **relative to the baseline** $\phi_0 = E[\hat f(X)]$. Change the baseline (different background dataset, or a single reference instance instead of the dataset average) and every $\phi_j$ changes too, even though they'll still sum to `prediction − baseline`. SHAP answers "how did this feature move the prediction *relative to* what we'd expect on average," not "what is this feature's inherent importance."

*SHAP 值衡量的是相对于 baseline 的偏移量，换一个 baseline，每个特征的 SHAP 值都会跟着变。*

---

## 3. "Sublinear": why exact Shapley values are (usually) not computed directly

*(Also ambiguous in the original notes — interpreted here as the computational-complexity side of SHAP.)*

Exact Shapley values require evaluating $val(S)$ over every subset of the other features — **O(2^M)** for M features, intractable beyond a handful. Two practical workarounds:

- **Sampling / Kernel SHAP**: sample a manageable number of coalitions and either average marginal contributions (the Monte-Carlo estimator above) or solve the weighted regression $L(\hat f, g, \pi_x)$ — turning an exponential problem into a tunable, polynomial-time approximation [1][2].
- **Model-specific exact algorithms (TreeSHAP)**: for tree ensembles, the tree structure allows *exact* SHAP values in low-order polynomial time (~O(T·L·D²)), no sampling needed.

So the practical story isn't literally "sublinear" — it's "exponential in theory, brought down to polynomial-time approximation or model-specific exact algorithms in practice." This also matches a LEAF finding (next section): SHAP stays deterministic only with few features, since more features push it into the sampling regime.

*精确算 Shapley 值是指数级 $O(2^M)$，特征一多就算不动，所以实际用采样近似（多项式时间）或针对树模型的精确算法（TreeSHAP）。*

---

## 4. What is LEAF?

**LEAF (Local Explanation Evaluation Framework)** evaluates the *quality* of local explanations from methods like LIME/SHAP, rather than assuming they're trustworthy by default [5]. It scores explanations on five axes:

- **Conciseness** – how many non-zero features are shown.
- **Local fidelity** – how well the local surrogate mimics the black box near the instance.
- **Local concordance** – whether the explanation's own output matches the black box's real prediction.
- **Reiteration similarity** – stability across repeated runs (Jaccard similarity of selected features) — directly measuring the non-determinism problem above.
- **Prescriptivity** – whether acting on the explanation would actually flip the model's decision.

Findings: LIME explanations are often unstable with weak local concordance (especially against non-linear classifiers); SHAP is deterministic only with few features — more features push it into the sampling-approximation regime, reintroducing variance [5].

*LEAF 不是新的解释方法，而是给 LIME/SHAP 的解释"打分"的评估框架，用五个指标衡量解释是否可信、稳定、可操作。*

---

## 5. What is a global surrogate model?

A **global surrogate model** is an interpretable model (shallow tree, linear model, etc.) trained to approximate a black box's *predictions* across the whole input space [6]:

1. Pick a dataset → 2. get black-box predictions on it → 3. choose an interpretable model class → 4. train it on (features → black-box predictions) → 5. measure fit → 6. interpret the surrogate as a stand-in for the black box.

Fit is typically measured with $R^2$:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}\big(f(x^{(i)}) - g(x^{(i)})\big)^2}{\sum_{i=1}^{n}\big(f(x^{(i)}) - \bar{f}\big)^2}
$$

Flexible and simple, but: it explains the *model's* behavior (not necessarily reality), there's no agreed "good enough" $R^2$ threshold, and a good average fit can still hide poor local fit in specific regions [6].

*全局代理模型是训练一个简单模型去模仿黑箱的预测结果（不是真实标签），用 $R^2$ 衡量模仿得好不好；整体拟合好不代表每个局部区域都拟合得好。*

---

## 6. What is PDP (Partial Dependence Plot)?

A **Partial Dependence Plot** shows the marginal effect of one or two features $x_S$ on the prediction, averaged over the rest ($C$) [7]:

$$
\hat{f}_S(x_S) = \mathbb{E}_{X_C}\big[\hat{f}(x_S, X_C)\big] = \int \hat{f}(x_S, X_C)\, d\mathbb{P}(X_C) \;\approx\; \frac{1}{n} \sum_{i=1}^n \hat{f}\big(x_S,\, x_C^{(i)}\big)
$$

Fix the feature(s) of interest at $x_S$, keep every other feature at its real value $x_C^{(i)}$ from each training row, average the predictions across all $n$ rows — that's one point on the curve; sweep $x_S$ for the full curve.

Two limitations noted directly in the source [7]:
- **Independence assumption**: the formula "treats the features in $C$ regardless of their correlation with features in $S$" — when $S$ and $C$ are correlated, it can create unrealistic combinations (e.g. a 2m-tall 50kg person), distorting the curve. (ICE and ALE plots are common companions/fixes.)
- **Hides heterogeneous effects**: since PDP shows only the *average* effect, opposite effects across subgroups can cancel out into a misleadingly flat curve — plot ICE curves alongside PDP to catch this.

*PDP 固定住关心的特征 $x_S$，其余特征保持数据集里的真实取值不变，对所有样本算一遍预测取平均，得到 $x_S$ 处的曲线高度。两个局限：假设特征间独立（相关特征会拼出不真实的组合），以及只看平均效应会掩盖不同子群体里方向相反的效应。*

---

## 7. What is the Explainable Boosting Machine (EBM)?

The **Explainable Boosting Machine (EBM)**, from Microsoft's InterpretML [8], is a **glass-box model** — interpretable *by design*, not explained after the fact like SHAP/LIME on a black box. It's a modern, cyclic-gradient-boosting take on a **Generalized Additive Model (GAM)**.

*EBM（可解释提升机）来自 Microsoft InterpretML，是"天生可解释"的模型，用现代 boosting 技术训练传统的广义可加模型（GAM），兼顾可解释性和准确率。*

### 7.1 Model formula

Without interactions, a standard GAM:

$$
g\big(\mathbb{E}[y]\big) = \beta_0 + \sum_{j=1}^{M} f_j(x_j)
$$

With EBM's automatic pairwise interaction detection (making it a **GA²M**):

$$
g\big(\mathbb{E}[y]\big) = \beta_0 + \sum_{j=1}^{M} f_j(x_j) + \sum_{(i,j) \in \mathcal{I}} f_{i,j}(x_i, x_j)
$$

$g$ is the link function (identity for regression, logit $\ln\frac{u}{1-u}$ for classification); each $f_j$ is a learned, arbitrary-shape function of one feature (a piecewise-constant lookup table over binned values), not a single coefficient; $\mathcal I$ is the small set of feature pairs selected as worth modeling jointly [8].

*不带交互项时就是标准 GAM——链接函数 $g$ 作用在预测上等于截距加每个特征各自的形状函数 $f_j$ 之和；每个 $f_j$ 是一整条曲线而不是一个系数。加上自动检测出的少量交互项 $f_{i,j}$ 后就叫 GA²M。*

### 7.2 Training procedure

EBM's defining trick is **cyclic, round-robin boosting**: rather than greedily boosting on all features jointly like a normal GBM (which entangles feature contributions), EBM updates **one feature's function $f_j$ at a time**, taking a very small step (very low learning rate) before moving to the next feature, cycling through many outer rounds — *"train on one feature at a time in round-robin fashion using a very low learning rate so that feature order does not matter"* [8]. This gives order-invariance and fairer credit-sharing among correlated features. EBM additionally trains multiple **bagged** copies of this process and averages the resulting shape functions, improving accuracy and yielding confidence intervals per curve. Pairwise interaction terms are chosen by a fast scoring procedure (**FAST**, GA²M's interaction-detection algorithm) that ranks candidate feature pairs by how much residual variance they explain beyond the additive model, keeping only the top pairs so the model stays visualizable [8].

*cyclic boosting 是 EBM 和普通 GBM 最大的区别——每轮只小步更新一个特征的 $f_j$，避免某个特征"抢跑"独吞相关特征的信号；配合 bagging 提升稳定性和准确率。交互项通过 FAST 算法打分挑选，只保留最重要的少数几对，保证模型依然可画图、可解释。*

### 7.3 Strengths and trade-offs

- **Accuracy**: comparable to random forests/GBMs — unusual for an inherently interpretable model.
- **Exact, not approximate, transparency**: unlike SHAP/LIME's post-hoc approximations, EBM's shape functions $f_j$ *are* the model — plotting $f_j(x_j)$ shows exactly (not approximately) how that feature affects predictions.
- **Slow to train, fast to serve**: many small cyclic + bagging rounds make training slower than a typical GBM, but inference is just table lookups and additions — fast and memory-efficient in production.

*EBM 的优势——准确率接近黑箱模型；解释是精确的而非近似的；训练慢但推理极快，适合线上部署。*

---

## Reference

1. Molnar, C. *Interpretable Machine Learning* — [SHAP chapter](https://christophm.github.io/interpretable-ml-book/shap.html)
2. Molnar, C. *Interpretable Machine Learning* — [Shapley Values chapter](https://christophm.github.io/interpretable-ml-book/shapley.html)
3. [Unstable explanations when no random seed is assigned in LIME `explain_instance` (GitHub issue)](https://github.com/marcotcr/lime/issues/119)
4. [DLIME: A Deterministic Local Interpretable Model-Agnostic Explanations Approach (GitHub)](https://github.com/rehmanzafar/dlime_experiments)
5. [To trust or not to trust an explanation: using LEAF to evaluate local linear XAI methods (PeerJ)](https://peerj.com/articles/cs-479/)
6. Molnar, C. *Interpretable Machine Learning* — [Global Surrogate Models chapter](https://christophm.github.io/interpretable-ml-book/global.html)
7. Molnar, C. *Interpretable Machine Learning* — [Partial Dependence Plot chapter](https://christophm.github.io/interpretable-ml-book/pdp.html); see also [scikit-learn PDP/ICE docs](https://scikit-learn.org/stable/modules/partial_dependence.html)
8. [Explainable Boosting Machine – InterpretML documentation](https://interpret.ml/docs/ebm.html); see also [GeeksforGeeks EBM overview](https://www.geeksforgeeks.org/machine-learning/explainable-boosting-machines-ebms/)
9. [LIME – 知乎 (zhuanlan.zhihu.com/p/85791430)](https://zhuanlan.zhihu.com/p/85791430)
10. Cooper, A. [A Non-Technical Guide to Interpreting SHAP Analyses](https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/) — original source of the waterfall plot example (house-price data, feature values, and SHAP contributions)
