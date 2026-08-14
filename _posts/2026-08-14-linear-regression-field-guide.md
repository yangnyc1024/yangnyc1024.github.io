---
title: "Linear Regression, All the Way Down"
subtitle: "The last model in the stack whose every layer you can actually see — 整个 stack 里最后一个每层都看得见的模型"
date: 2026-08-14 12:00:00 +0800
categories:
  - Statistics
  - Linear Models
  - Interview Prep
writing_category: algorithm
tags: [machine-learning, statistics, interview-prep, linear-models]
reading_time: "~30 min"
description: "A field guide to linear regression: the two-tier assumption structure, three derivations of OLS, why squared loss, closed-form vs. gradient descent, inference and diagnostics, and the bias-variance case for Ridge and Lasso."
---

Linear regression is the last model in the stack where you can see everything. Write down the estimator in closed form; prove it's unbiased; derive its uncertainty; diagnose exactly how it fails. No approximation anywhere. Every layer — probabilistic assumptions, estimation theory, optimization, inference, diagnostics, regularization — is open to inspection, and each one has a defensible right answer.

Nothing else you work with offers that. Which is why it's worth more time than its apparent difficulty suggests.

The usual treatment wastes that. Five assumptions, listed flat, recited, forgotten — *correct*, and completely inert. The content is in the **structure** behind those five: which ones buy you what, and what specifically breaks when each one fails. That structure is what this post is about.

> 这篇是双语的：正文英文，每节末尾有一段 **📌 中文小结** 收拢要点，方便跳读。
> 标记 **💬 In the room** 的段落，谈的是同一个知识点被追问时该怎么措辞。

---

## 1. The model is stranger than it looks

Start with the obvious:

$$y = X\beta + \epsilon$$

Linear regression models the relationship between a target $y$ and features $X$. Fine. But here's the first place people get tripped up:

> **"Linear" means linear in the parameters $\beta$ — not linear in $x$.**
> **「线性」指的是对参数 $\beta$ 线性，不是对特征 $x$ 线性。** 这是最常被用来试探「你是不是背的」的一个点。

This distinction is not pedantry. It completely changes what the model can do:

| Model | Linear? | Why | 中文说明 |
|---|---|---|---|
| $y = \beta_0 + \beta_1 x$ | ✅ | Obviously | 最基本形式 |
| $y = \beta_0 + \beta_1 x + \beta_2 x^2$ | ✅ | $x^2$ is just another feature | $x^2$ 只是一个新特征，对 $\beta$ 仍线性 |
| $y = \beta_0 + \beta_1\log x + \beta_2 x_1 x_2$ | ✅ | Feature transforms don't touch parameter linearity | 特征变换不影响参数线性 |
| $y = \beta_0 x^{\beta_1}$ | ❌ | $\beta_1$ sits in an exponent | 参数跑到指数上了 |
| $y = 1/(\beta_0 + \beta_1 x)$ | ❌ | Nonlinear in $\beta$ | 对 $\beta$ 非线性 |

So **polynomial regression is linear regression**. Splines are linear regression. Interaction terms are linear regression. The model class is far larger than the straight line people picture.

> **💬 In the room｜面试怎么答**
>
> 被问 *"Can linear regression capture nonlinear relationships?"* 时，可以直接说：
>
> *"Yes — through feature engineering. The model is linear in $\beta$, and $\beta$ is all it estimates. What it genuinely can't do is **learn** which transformation to apply; you have to supply it. That's the real limitation."*
>
> **为什么这样答**：多数人会答「不能，线性回归只能拟合直线」，这是错的。答「能」已经赢了一半；再补上「真正的局限是它不会自己发现该做什么变换」，说明你理解的是模型的**能力边界**，而不只是模型的样子。

### Two distinctions worth nailing down

People conflate these constantly, and both distinctions do real work later in this post.

**Noise vs. Error｜噪声 vs. 误差**

| | Source | Reducible? | Object |
|---|---|---|---|
| **Noise（噪声）** | Randomness in the data-generating process<br>数据生成过程本身的随机性 | ❌ Irreducible<br>不可消除 | The true $\epsilon$, variance $\sigma^2$ |
| **Error（误差）** | Gap between your model and the truth<br>模型与真实函数的差距 | ✅ Reducible — better model, more data<br>可通过换模型/加数据减少 | Bias² + Variance |

Which gives the decomposition everything else in this post depends on:

$$\mathbb{E}\big[(y - \hat f(x))^2\big] = \underbrace{\text{Bias}[\hat f]^2 + \text{Var}[\hat f]}_{\text{reducible}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

Hold onto this. It's the entire argument for regularization in §8.

> 这个分解要一直记到第 8 节。**正则化的全部理论动机就藏在这里**：既然我们真正想最小化的是总 MSE，那么主动引入一点 Bias 去换 Variance 的大幅下降，是完全划算的。

**Error vs. Residual｜误差 vs. 残差**

- **Error（误差）** $\epsilon_i = y_i - x_i^T\beta$ — measured against the *true, unknown* $\beta$. **Unobservable.**
- **Residual（残差）** $e_i = y_i - x_i^T\hat\beta$ — measured against your *estimated* $\hat\beta$. **Observable.**

Every diagnostic you'll ever run uses residuals to make inferences about errors you can never see.

> 所有诊断图（残差图、Q-Q 图）用的都是**残差**，而我们真正想检验的是**误差**是否满足假设。**永远在用可观测的量去推断不可观测的量** —— 这就是第 7 节诊断部分的本质。

> **📌 中文小结｜§1**
>
> - **「线性」= 对参数线性**。$x^2$、$\log x$、交互项都还在线性模型里；Polynomial Regression 本质就是 Linear Regression。
> - 线性回归**能**拟合非线性关系，但**不会自己发现**该做什么变换 —— 这才是它真正的局限。
> - **Noise 不可消除**（数据自带），**Error 可以消除**（模型造的）。
> - **Error 不可观测，Residual 可观测**；诊断的全部工作就是用后者推断前者。
> - 记住 `MSE = Bias² + Variance + σ²`，第 8 节的正则化全靠它。

---

## 2. The assumptions, in two tiers

Here's the thing almost nobody does: **separate the assumptions by what they buy you.**

Most candidates rattle off five assumptions as a flat list. But they're not a flat list — they're two distinct tiers, and knowing which is which is the highest-leverage fact in this entire post.

> **这一节是全文最重要的一节。** 多数候选人把五条假设当成平铺的清单背下来 —— 那个答案「正确但没有记忆点」。真正拉开差距的是**分层**：哪些假设保证点估计无偏，哪些只是为了推断有效。能分层，就说明你知道每条假设各自在换什么东西。

Before the list, one picture. It encodes three assumptions at once:

![Conditional distribution of y given x, with and without homoscedasticity](/assets/posts/linear-regression-field-guide/fig1-conditional-distribution.png)

Read the left panel: at every $x$, $y$ is drawn from a normal distribution. The **centres of those bells lie on the regression line** — that's linearity, and it's why the line *is* $\mathbb{E}[Y\vert X]$. The bells are **the same width everywhere** — that's homoscedasticity. Their shape is **normal** — that's assumption 5.

The right panel keeps the centres on the line but lets the widths grow. This is the picture to hold onto for the rest of the section, because it shows the single most important asymmetry in one glance.

> **一张图同时编码了三条假设：**
>
> | 图上看到的 | 对应假设 |
> |---|---|
> | 所有钟的**中心连成一条直线** | Linearity —— 回归线就是**条件期望** $\mathbb{E}[Y\vert X]$ |
> | 所有钟**一样宽** | Homoscedasticity（同方差） |
> | 形状是**钟形** | Normality of errors |
>
> **右图是关键**：钟的中心**仍然落在线上**，所以 $\hat\beta$ 依然**无偏**；但宽度在变，所以 $\text{Var}(\epsilon\vert X)\neq\sigma^2I$，**标准误算错了**。
>
> 「异方差不影响估计、只影响推断」这句话，看图比看公式快得多。

### Tier A — what you need for unbiased, consistent estimates

**第一层：保证点估计无偏、一致所需的假设**

| # | Assumption | Formally | Buys you｜换来什么 |
|---|---|---|---|
| 1 | **Linearity in parameters**<br>参数线性 | $y = X\beta + \epsilon$ | Correct specification<br>模型设定正确，无 misspecification |
| 2 | **Exogeneity**<br>外生性 ★ | $\mathbb{E}[\epsilon \vert X] = 0$ | **This is where unbiasedness comes from**<br>**无偏性的唯一来源**，最核心的一条 |
| 3 | **No perfect multicollinearity**<br>无完全多重共线性 | $\text{rank}(X) = p$ | $X^TX$ invertible → unique solution<br>保证可逆，闭式解存在且唯一 |
| 4 | **Spherical errors**<br>球形误差 | $\text{Var}(\epsilon\vert X) = \sigma^2 I$ | 拆成两条：**homoscedasticity（同方差）** + **no autocorrelation（无自相关）** |

Satisfy 1–4 and you get the **Gauss-Markov theorem**: OLS is **BLUE** — the Best Linear Unbiased Estimator.

But read those letters carefully, because this is where the theorem is usually over-read:

- **B**est = minimum variance — *but only within the L+U class*
- **L**inear = only estimators of the form $Ay$ are in the comparison
- **U**nbiased = **the moment you allow bias, Ridge can beat it on MSE**

> **BLUE 三个字母的限定含义，是本节第二重要的追问点。**
>
> Gauss-Markov 并没有说「OLS 是最好的估计量」，它说的是「OLS 在**线性 + 无偏**这个受限的比赛里最好」。一旦允许有偏，Ridge 就可能在 MSE 上赢它 —— **这句话就是正则化的理论入口**，第 8 节会回到这里。

### Tier B — what you need only for valid inference

**第二层：仅为了推断（标准误、置信区间、假设检验）有效所需的假设**

| # | Assumption | Formally |
|---|---|---|
| 5 | **Normality of errors**（误差正态性） | $\epsilon\vert X \sim \mathcal{N}(0, \sigma^2 I)$ |

That's it. One assumption, and it does nothing for your point estimates.

### Why normality doesn't buy unbiasedness

Just follow the algebra:

$$\mathbb{E}[\hat\beta\vert X] = \mathbb{E}\big[(X^TX)^{-1}X^T(X\beta + \epsilon)\big\vert X\big] = \beta + (X^TX)^{-1}X^T\underbrace{\mathbb{E}[\epsilon\vert X]}_{=\,0} = \beta$$

The only inputs are linearity and $\mathbb{E}[\epsilon\vert X]=0$. The error distribution never appears.

> **这条推导要能当场写出来。** 整个过程只用到了两件事：模型对参数线性、误差条件均值为 0。**误差的分布形状从头到尾没有出现过。** 所以误差服从 Uniform 也好、Student-t 也好，只要条件均值是 0，$\hat\beta$ 就是无偏的。

Consistency comes from the **LLN** ($\frac1n X^T\epsilon \to 0$), and the asymptotic normality of $\hat\beta$ comes from the **CLT**. Note the subtlety, because this is the sentence that lands:

> The CLT gives you a normal **sampling distribution of $\hat\beta$**. It does not require $\epsilon$ itself to be normal.
>
> **中心极限定理保证的是 $\hat\beta$ 这个估计量的抽样分布渐近正态，而不是要求误差 $\epsilon$ 本身正态。** 这两件事经常被混为一谈 —— 前者是关于估计量的，后者是关于数据的。

### So why assume normality at all?

Two reasons, both narrow:

1. **Exact finite-sample inference（精确小样本推断）.** Only under normal errors do the $t$ and $F$ statistics follow exact $t$ and $F$ distributions, making $p$-values and CIs exactly rather than approximately valid.
   > 只有误差正态时，$t$ 和 $F$ 统计量才**精确**服从 $t$/$F$ 分布。**大样本下靠 CLT 就够了，正态假设可以直接放弃。**
2. **Efficiency（有效性）.** Under normality, OLS isn't just BLUE — it attains the **Cramér-Rao lower bound**, making it **UMVUE**: minimum variance among *all* unbiased estimators, not just linear ones.
   > 正态假设下 OLS 从 BLUE 升级为 **UMVUE** —— 在**所有**无偏估计量中方差最小，不再限于线性那一类。

> **💬 In the room｜面试怎么答**
>
> 两层框架是这一节 ROI 最高的东西。开场就这么说：
>
> *"I'd separate these by what they buy. Linearity, zero conditional mean, and no perfect collinearity give me unbiasedness and consistency. Adding spherical errors gives me Gauss-Markov efficiency. Normality is a separate matter entirely — it's only about exact small-sample inference."*
>
> 说完就停，等对方追问。
>
> **为什么这样答**：这段话主动**标记出了知识的接缝在哪里**。面试官几乎一定会顺着其中一句往下问（最常见的是「为什么 normality 不影响无偏」），而你已经提前把问题引到了自己准备好的地方 —— 这是把面试节奏握在自己手里。

### When each assumption breaks

The follow-up is always *"what happens if X fails?"* Answer in three beats: **consequence → diagnosis → fix**（后果 → 诊断 → 修复）.

| Broken | Consequence｜后果 | Diagnosis｜诊断 | Fix｜修复 |
|---|---|---|---|
| **Exogeneity**<br>外生性 | **Biased AND inconsistent**<br>**有偏且不一致** —— 最严重，加数据也救不回来 | A reasoning problem, not a plot<br>靠理论判断：遗漏变量 / 反向因果 / 测量误差 / 样本选择 | IV / 2SLS, fixed effects, DiD, RCT<br>工具变量、固定效应、双重差分、实验 |
| **Homoscedasticity**<br>同方差 | Still **unbiased**, but **SEs are wrong**<br>估计仍无偏，但标准误算错 → $t$/$p$/CI 全失效 | Residuals-vs-fitted fans out<br>残差图呈喇叭口；Breusch-Pagan / White 检验 | **Robust (Huber-White) SEs**<br>稳健标准误（首选，一行代码）；WLS；对 $y$ 取 log |
| **No autocorrelation**<br>无自相关 | Same; SEs typically **understated**<br>同上，且标准误通常被低估 → 显著性虚高 | Durbin-Watson; ACF of residuals<br>残差自相关图 | Newey-West / clustered SEs, GLS, add lags<br>加滞后项 |
| **Normality**<br>正态性 | Small-sample inference invalid; large samples fine<br>只影响小样本推断 | Q-Q plot, Shapiro-Wilk | Lean on CLT, bootstrap CIs, robust regression |
| **Perfect collinearity**<br>完全共线 | $X^TX$ singular — **no unique solution**<br>解不唯一 | $\text{rank}(X) < p$; VIF $\to\infty$ | Drop features, Ridge, PCA |
| **Linearity**<br>线性 | Misspecification → **biased**<br>设定错误导致有偏 | Systematic curvature in residual plot; RESET test | Polynomial/interaction terms, splines, GAM |

Notice the asymmetry that matters most: **heteroscedasticity and autocorrelation only break your standard errors. Endogeneity breaks your estimates.**

> **这个不对称是本节的第三个得分点。** 异方差和自相关只毁掉标准误 —— 换成 robust SE，一行代码解决；内生性毁掉的是估计量本身 —— 那是**研究设计问题**，不是统计技术问题，需要工具变量或改实验设计。
>
> 把六种违背当成同等严重来回答的候选人，说明还没想透。

> **📌 中文小结｜§2**
>
> - **假设分两层**：Tier A（参数线性 + $E[\epsilon\vert X]=0$ + 无完全共线 + 球形误差）买的是**无偏、一致、BLUE**；Tier B（正态性）只买**小样本精确推断**。
> - **$E[\epsilon\vert X]=0$ 是无偏性的唯一来源** —— 推导里只用到它，误差分布形状全程没出现。
> - **CLT 保证的是 $\hat\beta$ 的抽样分布正态，不是 $\epsilon$ 正态**。这两件事别混。
> - **BLUE = 在「线性 + 无偏」这个受限比赛里最好**。允许有偏，Ridge 就能赢 → 正则化的入口。
> - 违背假设时按 **后果 → 诊断 → 修复** 三段答。
> - **只有内生性会让估计量有偏**，其余都只影响推断。这个不对称一定要说出来。

---

## 3. One estimator, three derivations

Here's what I find genuinely elegant about OLS: three completely different intellectual traditions converge on the same formula.

> 同一个 $\hat\beta$，三种完全不同的世界观都能推出来。**能一次讲全三条，是这道题的满分答案。**

### Derivation 1 — Least squares (the optimization view)

**路径一：最小二乘（优化视角）**

$$\hat\beta = \arg\min_\beta \|y - X\beta\|_2^2$$

Take the gradient and set it to zero:

$$\frac{\partial}{\partial\beta}\big[(y-X\beta)^T(y-X\beta)\big] = -2X^T(y - X\beta) = 0$$

$$\Longrightarrow \quad \boxed{X^TX\hat\beta = X^Ty \quad\Longrightarrow\quad \hat\beta = (X^TX)^{-1}X^Ty}$$

That's the **normal equation（正规方程）**. The Hessian is $2X^TX \succeq 0$, so the objective is convex and this stationary point is the global minimum.

> 二阶导 $2X^TX$ 半正定 → 目标函数**凸** → 驻点即全局最小。这句补充能省掉「你怎么知道这是最小值不是最大值」的追问。

For simple regression, worth memorizing because you may be asked to derive it on a whiteboard:

$$\hat\beta_1 = \frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2} = \frac{\text{Cov}(x,y)}{\text{Var}(x)}, \qquad \hat\beta_0 = \bar y - \hat\beta_1\bar x$$

> 一元的显式解值得单独记住，因为它把系数还原成了两个最基本的统计量之比。而 $\hat\beta_0 = \bar y - \hat\beta_1\bar x$ 有个直接的几何后果：**回归线必然穿过样本均值点 $(\bar x,\bar y)$**。

### Derivation 2 — Maximum likelihood (the statistical view)

**路径二：极大似然（统计视角）**

Every MLE derivation opens with the same question:

> **"What distribution does $Y\vert X$ follow?"**

And that question is the entire derivation. Once the distribution is named, everything downstream is mechanical — multiply, take logs, differentiate. The distribution is the only real choice being made.

> **这句问题本身就是方法。**
>
> 一旦定下 $Y\vert X$ 服从什么分布，后面全是机械操作：连乘、取 log、求导。**整条推导里唯一需要做判断的地方，就是这一句。**
>
> 所以真正体现懂不懂的，从来不是后面的代数，而是有没有先问这一句 —— 换个分布（比如 Laplace），同样的机械操作会带你走到完全不同的损失函数上。下一节就走这条路。

Assume $\epsilon_i \overset{iid}{\sim}\mathcal{N}(0,\sigma^2)$, so $y_i\vert x_i \sim \mathcal{N}(x_i^T\beta, \sigma^2)$:

$$p(y_i\vert x_i) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(y_i - x_i^T\beta)^2}{2\sigma^2}\right)$$

Multiply across i.i.d. samples, then take logs:

$$\ell(\beta) = \underbrace{-n\ln\sigma - \tfrac n2\ln(2\pi)}_{\text{constant in }\beta} \;-\; \frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - x_i^T\beta)^2$$

Maximizing $\ell$ is *identical* to minimizing the sum of squared residuals. **Under Gaussian noise, MLE and OLS are the same estimator.**

Go back to the left panel of the figure in §2 and this becomes almost physical: you're sliding the line until every observed point sits in the high-density part of its own bell. The position that maximizes that joint density is the OLS fit.

> 回头看 §2 那张图的左栏，MLE 的含义几乎是**物理的**：你在挪动这条线，让每个观测点都落进它头上那个钟的**高密度区**。挪到联合密度最大的位置，就是 OLS 解。
>
> 这比盯着 $\ell(\beta)$ 的公式直观得多。

The line I'd actually say:

> *"One comes from statistics, the other from optimization. They coincide precisely because the Gaussian log-density is quadratic in the residual."*

> **注意 "precisely because" 这个措辞在做什么。** 它标记出这个巧合是**有条件的**，不是天经地义的 —— 高斯对数密度关于残差恰好是二次式，取 log 之后才长成平方损失。换一个噪声分布，两者立刻分道扬镳（下一节就走这条路）。

One more detail: the MLE for $\sigma^2$ is $RSS/n$, which is **biased downward**. The unbiased estimator divides by degrees of freedom: $\hat\sigma^2 = RSS/(n-p-1)$.

> $\sigma^2$ 的 MLE 是 $RSS/n$，**有偏（低估）**；除以自由度 $n-p-1$ 才无偏。这就是后面 RSE 公式里分母为什么不是 $n$。

### Derivation 3 — Orthogonal projection (the geometric view)

**路径三：正交投影（几何视角）**

This is the one most candidates skip, and it's the cheapest way to sound like you actually understand the method.

> 这条路径最容易被跳过，但它解释的东西和前两条不一样：前两条告诉你 $\hat\beta$ **等于什么**，几何视角告诉你最小二乘**为什么是它**而不是别的准则。

OLS **projects $y$ orthogonally onto the column space of $X$**:

$$\hat y = X\hat\beta = \underbrace{X(X^TX)^{-1}X^T}_{H,\ \text{the hat matrix}}\,y = Hy$$

The hat matrix $H$（帽子矩阵）is a projection operator:

- **Idempotent（幂等）**: $H^2 = H$｜投影两次等于投影一次
- **Symmetric（对称）**: $H^T = H$
- $\text{tr}(H) = p+1$｜迹等于参数个数
- Residuals $e = (I-H)y$ satisfy $X^Te = 0$｜**残差与每一个特征都正交**
- $h_{ii}$, the diagonal entries, are **leverages（杠杆值）** — used in §7

Under this reading, the normal equation $X^T(y - X\hat\beta) = 0$ just says: *the residual is perpendicular to the column space.*

> **正规方程的几何含义就一句话：残差垂直于列空间。**
>
> 这也解释了最小二乘为什么不是一个随便选的准则 —— 它是**唯一**能让误差与你建模的所有东西都正交的选择。换句话说：残差里不再含有任何特征能解释的成分，信息已经榨干了。

> **📌 中文小结｜§3**
>
> - **三条路径**：最小二乘（求导令零）、MLE（先问 $Y\vert X$ 什么分布）、正交投影（几何）。能讲全三条是满分。
> - **正规方程**：$\hat\beta = (X^TX)^{-1}X^Ty$；凸性来自 Hessian $2X^TX\succeq0$。
> - 一元显式解 $\hat\beta_1 = \text{Cov}(x,y)/\text{Var}(x)$ —— 系数就是**协方差与方差之比**；由此 **回归线必过 $(\bar x,\bar y)$**。
> - **OLS = MLE 只在高斯噪声下成立**，原因是高斯对数密度关于残差是二次式。换噪声就不等价了。
> - $\sigma^2$ 的 MLE 有偏（除以 $n$），无偏要除自由度 $n-p-1$。
> - **几何本质：把 $y$ 正交投影到 $X$ 的列空间**；正规方程 = 残差垂直于列空间。

---

## 4. Why squared loss? And what happens if you change it

This question comes up constantly, and there's a shallow answer ("it's differentiable") and a deep one. Give the deep one.

> 这题必考。浅答案是「因为可导」，深答案有三层 —— 一定要给深的。

### Three independent justifications

**① Probabilistic｜概率视角.** Squared loss *is* the negative log-likelihood under Gaussian noise, up to a constant.

> 平方损失**就是**高斯噪声下的负对数似然。如果你相信误差近似高斯，那么平方损失不是一个「选择」，而是被**推导出来的必然结果**。

**② Optimization-theoretic｜优化视角.** It's strictly convex, $C^\infty$ smooth, and it's the only common loss with a **closed-form solution**. It also gives you the clean $\text{MSE} = \text{Bias}^2 + \text{Variance}$ decomposition — which, notably, *only* holds for squared loss.

> 严格凸、处处光滑、**唯一有闭式解**的常用损失。对比 L1 在 0 点不可导，需要次梯度方法。另外那个漂亮的 Bias-Variance 分解**只在平方损失下成立** —— 这点很多人不知道。

**③ Decision-theoretic｜决策视角.** This is the one to lead with:

$$\hat y^*(x) = \arg\min_c\ \mathbb{E}\big[(y-c)^2\,\big\vert\,X=x\big] = \boxed{\mathbb{E}[y\vert X=x]}$$

**Squared loss targets the conditional mean.** Linear regression isn't "fitting a line" — it's *estimating a conditional expectation*.

> **三条理由里，这条的层级最高。**
>
> 平方损失的最优预测是**条件期望**。所以线性回归的本质根本不是「拟合一条直线」，而是**估计一个条件期望** $\mathbb{E}[y\vert x]$ —— 直线只是实现方式。这个视角一转过来，下一小节的追问就变得非常自然。

### The follow-up: heavy-tailed noise

**追问：如果噪声是重尾的呢？**

Suppose $\epsilon_i \sim \text{Laplace}(0,b)$, so $p(\epsilon_i) \propto \exp(-\vert\epsilon_i\vert/b)$. Run the same MLE machinery:

$$\ell(\beta) = C - \frac1b\sum_{i=1}^n\big\vert y_i - x_i^T\beta\big\vert$$

Three things change, and you should name all three:

1. **The loss becomes L1** instead of L2.
   > 损失从 L2 变成 **L1（绝对值损失 / MAE）**。
2. **It's robust.** The L1 gradient is constant at $\pm1$, so any single point contributes a bounded amount. Squared loss amplifies outliers quadratically.
   > **稳健性**：L1 的梯度恒为 $\pm1$，单个点的贡献**有界**；L2 会平方放大 outlier，一个极端点就能把整条回归线拖歪。
3. **The target statistic changes.** L1 estimates the **conditional median**.
   > **目标统计量变了**：L1 估计的是**条件中位数**，不是条件均值。

That third point is the real insight. Choosing a loss isn't a numerical preference — **it's a statement about which functional of the conditional distribution you want to estimate.**

> **第三点才是真正的洞察。**
>
> 选损失函数不是在选「哪个数值上好算」，而是在**声明你想估计条件分布的哪一个泛函** —— 均值？中位数？还是某个分位数？
>
> 换句话说，损失函数是个**统计学决定**，不是工程细节。

### The loss family, organized

| Loss | Form | Implied noise｜隐含噪声 | Targets｜目标统计量 | Outliers | Smooth? |
|---|---|---|---|---|---|
| **Squared (L2)** | $(y-\hat y)^2$ | Gaussian | Conditional **mean**｜条件均值 | Sensitive ❌ | $C^\infty$ ✅ |
| **Absolute (L1)** | $\vert y-\hat y\vert$ | Laplace | Conditional **median**｜条件中位数 | Robust ✅ | Kink at 0 |
| **Huber** | quadratic inside $\delta$, linear outside | Gaussian core + heavy tails | Between the two | Robust ✅ | $C^1$ ✅ |
| **Quantile (pinball)** | $\max(\tau r, (\tau-1)r)$ | Asymmetric Laplace | The $\tau$-th **quantile**｜第 $\tau$ 分位数 | Robust ✅ | Kink at 0 |
| **Log-cosh** | $\ln\cosh(r)$ | — | ≈ mean | Robust ✅ | $C^2$ ✅ |

> **💬 In the room｜面试怎么答**
>
> 被问 *"L2 is fragile to outliers but L1 is awkward to optimize — what would you do?"*：
>
> *"I'd use **Huber loss** — quadratic near zero to preserve smoothness and statistical efficiency, linear in the tails for robustness, with $\delta$ set from a residual quantile."*
>
> **为什么这样答**：Huber 是「知道理论」和「真的做过项目」的分水岭。小残差用二次保住光滑性和统计效率，大残差切成线性保住稳健性 —— 而且 $\delta$ 该怎么定（按残差分位数自适应）也顺带答了。
>
> 可以主动补的一点：如果需要的是**预测区间**而不是点预测，直接上 **quantile regression**，拟合 $\tau=0.05$ 和 $\tau=0.95$ 就得到 90% 区间，**不需要任何分布假设**。这个补充经常能让对方眼前一亮。

The compressed version worth memorizing:

> **Loss ← noise distribution. Regularizer ← parameter prior.**
> **损失函数来自噪声分布假设，正则化项来自参数先验假设。**
>
> 这两句话能串起本文 80% 的推导。第 8 节会把后半句补完。

> **📌 中文小结｜§4**
>
> - **三个理由**：① 高斯噪声下的负对数似然（MLE）；② 严格凸 + 光滑 + 唯一有闭式解；③ **minimizer 是条件均值**（这条最有高度，建议先说）。
> - 线性回归的本质是**估计条件期望** $\mathbb{E}[y\vert x]$，不是「拟合直线」。
> - **重尾噪声（Laplace）→ L1 损失 → 条件中位数**，且对 outlier 稳健（梯度恒为 $\pm1$，贡献有界）。
> - **选损失 = 声明你要估计条件分布的哪个泛函**。这是这题最深的一层。
> - 工程上的标准答案是 **Huber**；要预测区间就用 **quantile regression**。
> - 记住 `Loss ← 噪声分布`，`Regularizer ← 参数先验`。

---

## 5. Solving it, for real

The closed form is beautiful. It's also, frequently, the wrong thing to use.

> 闭式解很漂亮，但在真实场景里经常是错误选择。常见追问：*"数据有 1 亿行 / 100 万个特征，你还用正规方程吗？"*

| | **Normal equation** | **Gradient descent / SGD** | **QR / SVD** |
|---|---|---|---|
| Form | $(X^TX)^{-1}X^Ty$ | $\beta \leftarrow \beta - \eta X^T(X\beta - y)/n$ | $R^{-1}Q^Ty$ |
| Complexity | $O(np^2 + p^3)$ | $O(np)$ per iteration | $O(np^2)$ |
| Learning rate?｜需调学习率 | No | **Yes** | No |
| Feature scaling?｜需标准化 | **No** | **Yes, critically** | No |
| Large $p$ ($>10^4$) | ❌ $p^3$ blows up | ✅ | ❌ |
| Large $n$ | ❌ $X^TX$ won't fit | ✅ mini-batch | ❌ |
| Ill-conditioned $X$｜病态矩阵 | ❌ falls apart | ✅ still runs | ✅ **most stable** |

Practical rules:

- Small $p$ (say $\lesssim 10^3$), moderate $n$ → closed form. But **never actually invert the matrix.** Use `np.linalg.solve` (LU) or `lstsq` (SVD).
  > $p$ 小、$n$ 适中 → 闭式解。但**永远不要真的求逆** —— 显式求逆既慢又不稳定，用 `solve` 或 `lstsq`，数值稳定性差一个数量级。
- Large $n$ or $p$ → SGD, mini-batch, or L-BFGS.
- Ill-conditioned design → SVD pseudo-inverse, or just add Ridge, which fixes conditioning as a side effect (§8).
  > 矩阵病态 → SVD 求伪逆，或者直接上 Ridge —— **Ridge 改善条件数是顺带的副作用**，第 8 节会讲清楚原理。

> **Why does GD need feature scaling but the closed form doesn't?**
>
> With features on wildly different scales, the loss contours become extremely elongated ellipses. The gradient points nearly perpendicular to the direction you actually need to travel, so you zig-zag and converge slowly. Standardizing makes the contours near-circular.
>
> The closed form is invariant: rescaling a feature just rescales its coefficient inversely, and the fitted values are identical.
>
> **中文解释**：特征尺度差异大时，损失函数的等高线是**极度拉长的椭圆**，梯度方向几乎垂直于真正该走的方向，于是 Z 字形震荡、收敛极慢。标准化后等高线接近圆形，梯度直指最优点。
>
> 而**闭式解对缩放是不变的** —— 缩放一个特征只会把对应系数反向等比缩放，拟合值完全一样。所以：**GD 要标准化，正则化要标准化，纯 OLS 不用。**

> **📌 中文小结｜§5**
>
> - **$p\lesssim10^3$ 且 $n$ 适中 → 闭式解**；$n$ 或 $p$ 很大 → **SGD / mini-batch / L-BFGS**；矩阵病态 → **SVD 或 Ridge**。
> - **永远不要用 `np.linalg.inv`**，用 `solve`（LU）或 `lstsq`（SVD）。
> - **GD 必须标准化，闭式解不用** —— 因为闭式解对特征缩放不变，而 GD 会被拉长的等高线拖垮。
> - 复杂度记住：闭式解 $O(np^2+p^3)$，GD 每轮 $O(np)$。

---

## 6. Reading the output

You've fit the model. Now: **is this coefficient real, or did noise hand it to you?**

> 这一层回答的问题是：**这个系数是真的，还是噪声碰巧凑出来的？**

Under assumptions 1–5:

$$\hat\beta \sim \mathcal{N}\big(\beta,\ \sigma^2(X^TX)^{-1}\big), \qquad \text{SE}(\hat\beta_j) = \hat\sigma\sqrt{\big[(X^TX)^{-1}\big]_{jj}}$$

### Testing one coefficient｜单系数检验

Under $H_0: \beta_j = 0$,

$$t = \frac{\hat\beta_j}{\text{SE}(\hat\beta_j)} \;\sim\; t_{\,n-p-1}$$

and the 95% CI is $\hat\beta_j \pm t_{\alpha/2}\cdot\text{SE}(\hat\beta_j)$.

> **原假设 $H_0:\beta_j=0$** 的含义是「该特征对 $y$ 没有线性影响」。注意一个常被忽略的等价关系：**置信区间不包含 0 $\iff$ $p<0.05$** —— 它们是同一句话的两种写法，不是两个独立的判据。

### Four ways people misread a p-value

**p-value 的四个经典误读**

| Wrong | Right |
|---|---|
| "$p$ is the probability $H_0$ is true"<br>「$p$ 是原假设为真的概率」 | $p$ is the probability of data **this extreme *given* $H_0$**<br>是**给定 $H_0$ 为真时**看到这么极端数据的概率 —— **条件方向反了** |
| "Small $p$ means a big effect"<br>「$p$ 小 = 效应大」 | With enough data, **trivially small effects become significant**<br>样本量够大时，**毫无实际意义的微小效应也会显著**。必须同时报 effect size |
| "$p > 0.05$ means no effect"<br>「$p>0.05$ = 没有效应」 | It means **insufficient evidence** — often just low power<br>只是**证据不足**，往往是样本量不够（power 低） |
| Testing 20 features uncorrected<br>测 20 个特征不做校正 | At $\alpha=0.05$ you **expect one false positive**<br>期望会有 1 个假阳性。用 **Bonferroni** 或 **FDR (Benjamini-Hochberg)** |

### Testing the whole model｜整体检验

$$F = \frac{(TSS - RSS)/p}{RSS/(n-p-1)} \;\sim\; F_{\,p,\,n-p-1}$$

Why bother, if you already have per-coefficient $t$-tests? Because with many features, some will look significant by chance — the $F$-test is a **global** test immune to that multiplicity.

> **为什么有了 $t$ 还要 $F$？** 因为 $p$ 很大时，即使所有特征都无关，也会有约 5% 的特征**偶然显著**（多重比较问题）。$F$ 检验问的是「所有特征联合起来到底有没有用」，是**全局检验**，不受这个污染。

And there's a diagnostic pattern hiding in the relationship between them:

> **$F$ is significant but no individual $t$ is** → that's the textbook signature of **multicollinearity（多重共线性）**.
>
> **$F$ 显著但没有任何单个 $t$ 显著** —— 这是共线性的教科书信号。含义是：这些特征**联合起来**解释力很强，但**单独拎出来**谁也说不清自己的贡献。第 8 节会讲为什么。

> **📌 中文小结｜§6**
>
> - $\hat\beta \sim \mathcal{N}(\beta,\ \sigma^2(X^TX)^{-1})$，$\text{SE}$ 是协方差矩阵对角元开方。
> - **$t$ 检验单系数，$F$ 检验整体**；$F$ 的价值在于免疫多重比较。
> - **CI 不含 0 $\iff$ $p<0.05$**，是同一件事。
> - **p-value 四误读**：① 条件方向反了；② $p$ 小 ≠ 效应大（大样本下微小效应也显著）；③ $p>0.05$ 只是证据不足；④ 多重比较必须校正。
> - **$F$ 显著 + 所有 $t$ 不显著 = 共线性的典型信号**。

---

## 7. When it goes wrong

### Evaluation metrics｜评估指标

| Metric | Form | Reads as｜含义 | Watch out｜陷阱 |
|---|---|---|---|
| **RSS** | $\sum(y_i-\hat y_i)^2$ | Total squared error | 随 $n$ 增长，不可跨数据集比 |
| **RSE** | $\sqrt{RSS/(n-p-1)}$ | Typical residual size, **in $y$'s units**<br>残差的典型大小，与 $y$ 同量纲 | 依赖 $y$ 的尺度 |
| **RMSE** | $\sqrt{RSS/n}$ | Same idea | 对 outlier 敏感 |
| **MAE** | $\frac1n\sum\vert y_i-\hat y_i\vert$ | Robust alternative｜稳健版本 | 不可导 |
| **MAPE** | $\frac1n\sum\vert(y_i-\hat y_i)/y_i\vert$ | Scale-free relative error｜相对误差，跨尺度可比 | $y_i\approx0$ 时爆炸；**不对称** —— 对高估的惩罚重于低估 |
| **$R^2$** | $1 - RSS/TSS$ | Share of variance explained<br>$y$ 的变异中被解释的比例 | 见下 ⚠️ |
| **Adjusted $R^2$** | $1-\frac{RSS/(n-p-1)}{TSS/(n-1)}$ | Penalizes feature count｜惩罚特征数 | 仍弱于 CV |
| **AIC / BIC** | $2p - 2\ell$ / $p\ln n - 2\ell$ | Fit + complexity penalty | BIC 惩罚更重 → 倾向更小的模型 |

The ANOVA identity behind $R^2$:

$$\underbrace{\textstyle\sum(y_i-\bar y)^2}_{TSS} = \underbrace{\textstyle\sum(\hat y_i-\bar y)^2}_{ESS} + \underbrace{\textstyle\sum(y_i-\hat y_i)^2}_{RSS}$$

> 这个分解**只在模型含截距项时成立**（因为它依赖 $\sum e_i = 0$）。不带截距拟合，$R^2$ 就不再是你以为的那个意思了 —— 这是个很少有人知道的细节。

### Four traps in $R^2$｜$R^2$ 的四个陷阱

1. **Adding any feature — including pure noise — never decreases $R^2$.** So you cannot use it for model selection.
   > **加任何特征（哪怕是纯随机噪声）$R^2$ 都不会下降**，所以**绝对不能用它选模型**。要用 Adjusted $R^2$ / AIC / BIC，最终以**交叉验证**为准。
2. **$R^2$ can go negative on a test set** — worse than predicting the training mean.
   > 测试集上 $R^2$ **可以为负**，意思是你还不如直接预测训练集均值。
3. In simple regression $R^2 = r^2$; in multiple regression $R^2 = \text{corr}(y,\hat y)^2$.
   > 一元回归里 $R^2$ 等于相关系数的**平方**；多元回归里等于 $\text{corr}(y,\hat y)^2$。
4. **High $R^2$ doesn't mean a good model.** **Anscombe's quartet** — four datasets with identical means, variances, correlations, regression lines, and $R^2$, but completely different shapes.
   > **Anscombe's Quartet**：四组数据的均值、方差、相关系数、回归线、$R^2$ 全部相同，但形态完全不同 —— 一组真的线性、一组是曲线、一组只有一个 outlier 在作怪、一组几乎退化。
   >
   > **所以：永远要画残差图。** 这也正是下一小节存在的理由。

### The four diagnostic plots｜残差诊断四件套

When asked *"how would you validate the assumptions?"*, this is the answer.

> 被问「你怎么验证假设成立」时，这四张图就是标准答案。

| Plot | Axes | Checks｜检验什么 | Bad sign｜异常信号 |
|---|---|---|---|
| **Residuals vs. Fitted** | $\hat y$ vs $e$ | Linearity + homoscedasticity<br>线性性 + 同方差 | 系统性弯曲 → 非线性；喇叭口 → 异方差 |
| **Q-Q plot** | Theoretical vs. sample quantiles | Normality｜正态性 | 两端偏离直线 → 重尾/偏态 |
| **Scale-Location** | $\hat y$ vs $\sqrt{\vert\text{std. resid}\vert}$ | Heteroscedasticity (more sensitive)<br>异方差（更敏感） | 出现任何趋势 |
| **Residuals vs. Leverage** | $h_{ii}$ vs std. residual + Cook's D | Influential points｜强影响点 | 落在 $D=0.5$ 等高线之外 |

Add an **ACF plot of residuals** for time series.｜时序数据再加**残差自相关图**。

### Outlier ≠ leverage ≠ influence

**离群点 ≠ 高杠杆点 ≠ 强影响点** —— 这三个概念被混用得最厉害。

| Concept | Definition | Measured by | Automatically bad?｜一定有害吗 |
|---|---|---|---|
| **Outlier**（离群点） | Unusual in $y$ — large residual<br>$y$ 方向异常，残差大 | Std. residual $\vert r_i\vert > 3$ | No — could just be noise<br>不一定，可能只是噪声大 |
| **High leverage**（高杠杆） | Unusual in $x$ — extreme features<br>$x$ 方向异常，特征极端 | $h_{ii} > 2(p+1)/n$ | No — if it sits on the line it *stabilizes* the fit<br>不一定，如果它正好落在回归线上，反而让模型更稳 |
| **Influential point**（强影响点） | **Removing it moves the coefficients**<br>**删掉后系数会明显改变** | **Cook's distance** $D_i$ | **Yes — investigate**｜是，必须调查 |

$$D_i = \frac{r_i^2}{p+1}\cdot\frac{h_{ii}}{1-h_{ii}}$$

Read that formula out loud and the intuition falls out: **influence = large residual × high leverage.**

> **把这个公式读出来，直觉就出来了：影响力 = 大残差 × 高杠杆。**
>
> 单独一项高都还能忍 —— 一个残差大但杠杆低的点，撬不动回归线；一个杠杆高但正好落在线上的点，反而稳定模型。**是两者的乘积才致命。**

> **💬 In the room｜面试怎么答**
>
> 被问「发现了强影响点怎么办」，**千万不要答「删掉」**。正确的回答分三层：
>
> *"First I'd check whether it's a **data error** — if so, fix it. If it's a genuine observation, I wouldn't silently delete it. I'd report results with and without it, or switch to robust regression, or ask whether I'm missing a feature that would **explain** it."*
>
> **为什么这样答**：随手删数据是数据科学里最容易被质疑的操作 —— 它意味着你在让数据迁就模型。**一个极端观测往往是信息，不是污染**。能说出「是不是缺了某个能解释它的特征」，说明你会往数据生成过程去想，而不只是在做统计操作。

> **📌 中文小结｜§7**
>
> - **$R^2$ 四陷阱**：① 加特征永不下降，不能用来选模型；② 测试集上可为负；③ 一元回归中等于 $r^2$；④ Anscombe's Quartet —— $R^2$ 相同，形态可以天差地别。
> - **ANOVA 分解只在含截距时成立**。
> - **诊断四件套**：Residuals-vs-Fitted（线性 + 同方差）、Q-Q（正态）、Scale-Location（异方差更敏感）、Residuals-vs-Leverage + Cook's D（影响点）。时序加 ACF。
> - **Outlier（$y$ 异常）≠ Leverage（$x$ 异常）≠ Influence（删掉会改变系数）**。
> - **Cook's D $\propto r_i^2 \times \frac{h_{ii}}{1-h_{ii}}$ —— 影响力是残差和杠杆的乘积**，单独一项高都不致命。
> - 发现影响点**不要直接删**：先查是否录入错误 → 报告有无该点两套结果 → 或换稳健回归 → 或想想是不是缺特征。

---

## 8. Collinearity, and the bias-variance bargain

### The collinearity problem｜共线性问题

**Multicollinearity（多重共线性）** is when features are highly linearly related, making $X^TX$ *nearly* singular.

The key facts, and the one that surprises people:

- $\hat\beta$ is **still unbiased**｜估计量**仍然无偏**
- But $\text{Var}(\hat\beta) = \sigma^2(X^TX)^{-1}$ **inflates dramatically** → unstable coefficients, signs that flip with tiny data changes, insignificant $t$'s alongside a significant $F$
  > 但方差**急剧膨胀** → 系数极不稳定、数据微小变动就能让符号翻转、单个 $t$ 不显著而 $F$ 显著
- **Prediction accuracy is essentially unaffected**｜**预测精度基本不受影响**

> **最后一点是最该先说的，也是最反直觉的。**
>
> 共线性是**可解释性问题，不是预测问题**。如果你的目标是预测，很多时候可以直接无视它；如果你要做归因、要向业务方解释「哪个特征贡献了多少」，那就必须处理。
>
> **判断原则**：目标是 prediction → 可以不管；目标是 inference / attribution → 必须处理。

Diagnose with the **variance inflation factor（方差膨胀因子）**:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ comes from regressing $x_j$ on all the other features. Rules of thumb: $>5$ is worth a look, $>10$ is serious.

> $R_j^2$ 是「用其他所有特征去回归 $x_j$」得到的 $R^2$ —— 如果其他特征能把 $x_j$ 解释得很好，说明它是冗余的。经验阈值：**VIF > 5 警戒，> 10 严重**。也可以看 $X^TX$ 的**条件数** $\kappa=\sqrt{\lambda_{\max}/\lambda_{\min}}$。

### Why regularize at all｜为什么需要正则化

Come back to Gauss-Markov. OLS has minimum variance **among unbiased linear estimators**. But we don't actually care about unbiasedness — we care about **total error**:

$$\text{MSE} = \text{Bias}^2 + \text{Variance}$$

> **Accept a little bias, buy a large reduction in variance, and total MSE goes down.**
>
> **主动接受一点偏差，换取方差的大幅下降，总 MSE 反而更低。**
>
> 这就是正则化的**全部**理论动机。回头看第 2 节的 BLUE —— Gauss-Markov 只保证 OLS 在「无偏」这个约束下最优，而**我们从来就没真的需要无偏**。一旦松开这个约束，就有更好的估计量在等着。

Applies whenever $p > n$（OLS 无唯一解）, collinearity is severe（共线性严重）, you have many features and suspect most are useless（特征多且大部分无用）, or you're overfitting（过拟合）.

### Ridge (L2)

$$\hat\beta_{\text{ridge}} = \arg\min_\beta \|y-X\beta\|_2^2 + \lambda\|\beta\|_2^2 \;\Longrightarrow\; \boxed{(X^TX + \lambda I)^{-1}X^Ty}$$

Three properties worth being able to state:

**1. It always has a unique solution.** $X^TX$ is PSD; adding $\lambda I$ with $\lambda>0$ makes it positive definite, hence invertible. **This works even when $p > n$.**

> $X^TX$ 是半正定的，加上 $\lambda I$（$\lambda>0$）后变成**正定**，必然可逆。**所以即使 $p>n$ 也有唯一解** —— 这是 Ridge 相对 OLS 最实际的优势。

**2. The SVD view shows you *where* it shrinks.** With $X = UDV^T$:

$$X\hat\beta_{\text{ridge}} = \sum_{j=1}^p u_j\,\frac{d_j^2}{d_j^2 + \lambda}\,u_j^Ty$$

Each principal direction is shrunk by $\frac{d_j^2}{d_j^2+\lambda}$. Small $d_j$ — the **low-variance directions** — get shrunk hardest. And low-variance directions are *precisely* where collinearity creates instability.

![Ridge shrinkage factor as a function of singular value](/assets/posts/linear-regression-field-guide/fig4-ridge-svd-shrinkage.png)

Plotted, the shrinkage factor is an S-curve in $d_j$. Directions with large singular values — where the data actually carries information — pass through nearly untouched. Directions with small $d_j$ get crushed. Ridge isn't a blunt instrument; it's aimed at the directions the design matrix is least sure about.

> 画出来是一条关于 $d_j$ 的 S 形曲线：**大 $d_j$ 的方向几乎原样通过，小 $d_j$ 的方向被压到接近 0**。$\lambda$ 越大，整条曲线整体下沉，但形状不变 —— 打击重点始终在左边。

> **这是本节最漂亮的一个事实。**
>
> Ridge 在每个主成分方向上按 $\frac{d_j^2}{d_j^2+\lambda}$ 收缩。$d_j$ 越小（数据在该方向上的方差越小），收缩得**越狠**。而**低方差方向恰恰就是共线性造成不稳定的方向** —— 所以 Ridge 不是无差别地压制所有系数，它是在**精准打击数据中信息量最少、最不可靠的那些方向**。
>
> 换个说法：$(X^TX+\lambda I)^{-1}$ 这个公式看不出 Ridge 在做什么，SVD 分解才看得出来。

**3. Under an orthonormal design it's proportional shrinkage:** $\hat\beta_j^{\text{ridge}} = \hat\beta_j^{\text{OLS}}/(1+\lambda)$. Everything shrinks toward zero; **nothing ever reaches it.**

> 正交设计下就是**等比例缩小**：所有系数除以 $(1+\lambda)$。**趋近于 0，但永远到不了 0** —— 这正是它和 Lasso 的分水岭。

### Lasso (L1)

$$\hat\beta_{\text{lasso}} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda\|\beta\|_1$$

No closed form — $\vert\cdot\vert$ isn't differentiable at zero — so you use **coordinate descent** or **LARS**.

> 因为绝对值在 0 点不可导，**没有闭式解**，要用**坐标下降**或 **LARS** 求解。

But under an orthonormal design, the solution is clean and extremely revealing:

$$\hat\beta_j^{\text{lasso}} = \text{sign}(\hat\beta_j^{\text{OLS}})\cdot\big(\vert\hat\beta_j^{\text{OLS}}\vert - \lambda\big)_+$$

This is **soft-thresholding（软阈值）**, and it *is* the answer to why Lasso is sparse.

![Soft-thresholding versus proportional shrinkage](/assets/posts/linear-regression-field-guide/fig3-soft-thresholding.png)

Put the two penalties on the same axes and the difference stops being subtle. Ridge is a straight line through the origin — it tilts toward zero and stays there. Lasso has a **flat segment pinned to zero**, exactly as wide as $2\lambda$. Any coefficient landing in that band comes out as exactly zero, not merely small.

> **两条线并排一放，区别就不再微妙了。**
>
> Ridge 是一条**过原点的斜线** —— 整体压向 0，但永远碰不到 0。
> Lasso 中间有一段**平压在 0 上的线段**，宽度正好是 $2\lambda$ —— 落进这个区间的系数出来就是**精确的 0**，不是「很小」。

> **这个公式本身就是「为什么 Lasso 稀疏」的答案**：所有绝对值小于 $\lambda$ 的系数被**直接推到 0**，而不是等比缩小。对比 Ridge 的「除以 $1+\lambda$」，差别一目了然。

### Why L1 is sparse and L2 isn't — three ways

**为什么 L1 稀疏而 L2 不稀疏 —— 三种解释**

The three aren't redundant — each explains a different layer.｜三套解释不是冗余，它们各自解释了不同的层面。

**① Geometric｜几何解释（最直观，可以在白板上画）**

![L1 and L2 constraint regions with loss contours](/assets/posts/linear-regression-field-guide/fig2-l1-l2-constraint.png)

Write it as a constrained problem: minimize $\|y-X\beta\|^2$ subject to $\|\beta\|_1 \le t$. The L1 constraint region is a **diamond with corners on the axes**; the L2 region is a smooth ball. The loss contours expand until they first touch the region — and a contour is quite likely to touch a **corner**, where some coordinates are exactly zero.

> 写成约束形式后：**L1 的可行域是菱形，尖角正好落在坐标轴上**；L2 的可行域是圆球，处处光滑。
>
> 损失函数的等高线（椭圆）从最优点向外扩张，**第一次碰到可行域的地方就是解**。椭圆碰到**菱形尖角**的概率很高，而尖角在坐标轴上 → 某些坐标恰好为 0 → **稀疏**。碰到圆球只会落在光滑边界上 → 所有坐标一般都非零。

**② Subgradient｜次梯度解释（最本质）**

Look at the penalty's derivative as $\beta_j\to0$:
- L2: $\frac{\partial}{\partial\beta_j}\beta_j^2 = 2\beta_j \to 0$ — the closer to zero you get, **the weaker the push**.
- L1: $\partial\vert\beta_j\vert = \pm1$ — a **constant push**, regardless of how close you are.

> **L2 的推力会随着系数接近 0 而消失**（$2\beta_j\to0$），所以它只能无限逼近、永远到不了。
> **L1 的推力恒为 $\pm1$，不管多接近 0 都一样大** —— 只要它超过数据的拟合梯度，就能把系数**牢牢钉死在 0**。
>
> 这是三套解释里最本质的一套，因为它解释的是**机制**而不是**现象**。

**③ Bayesian｜贝叶斯解释（最有理论深度）**

Regularization is MAP estimation with a prior:

$$\hat\beta_{\text{MAP}} = \arg\max_\beta \underbrace{\log p(y\vert X,\beta)}_{\text{likelihood}\ \to\ \text{loss}} + \underbrace{\log p(\beta)}_{\text{prior}\ \to\ \text{penalty}}$$

| Prior | Density | $-\log p(\beta)$ | Gives you |
|---|---|---|---|
| **Gaussian** $\mathcal{N}(0,\tau^2)$ | $\propto e^{-\beta^2/2\tau^2}$ | $\propto \beta^2$ | **Ridge** |
| **Laplace** $\text{Lap}(0,b)$ | $\propto e^{-\vert\beta\vert/b}$ | $\propto \vert\beta\vert$ | **Lasso** |

> **正则化 = 给参数加先验后做 MAP 估计。**
>
> Laplace 密度在 0 处有一个**尖峰（spike）** —— 它字面意义上就在表达「大部分系数本来就是 0」这个信念。而 Gaussian 在 0 处是光滑的，它只说「系数应该小」，从没说过「应该是 0」。
>
> **先验的形状决定了解的形状。**

And now the symmetry from §4 closes:

> **Loss ← noise distribution. Regularizer ← parameter prior.**
> Gaussian noise → L2 loss. Laplace noise → L1 loss.
> Gaussian prior → L2 penalty. Laplace prior → L1 penalty.
>
> **完整的对称结构：**
> **损失函数来自噪声分布假设** —— 高斯噪声给你 L2 损失，Laplace 噪声给你 L1 损失。
> **正则化项来自参数先验假设** —— 高斯先验给你 L2 惩罚，Laplace 先验给你 L1 惩罚。
>
> 这个对称一旦看清，线性回归就从**一堆各自为政的技巧**变成了**一个连贯的概率对象**：噪声假设决定你怎么惩罚残差，先验假设决定你怎么惩罚参数，两边用的是同一套逻辑。

### Elastic Net

$$\min_\beta \|y-X\beta\|_2^2 + \lambda\big(\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2\big)$$

Fixes two known Lasso failure modes:

1. When $p > n$, Lasso can select **at most $n$ features**.
   > $p>n$ 时，Lasso **最多只能选出 $n$ 个特征**。
2. Given a **group of correlated features**, Lasso arbitrarily keeps one and zeros the rest — and which one it keeps is unstable across resamples. The L2 term restores a **grouping effect（分组效应）**.
   > 面对一组高度相关的特征，Lasso 会**随机留下一个、其余全部归零**，而且换一批数据留下的可能是另一个 —— **极不稳定**。L2 项带来**分组效应**：相关特征被**一起保留、一起收缩**。

### Ridge vs. Lasso, side by side

| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| Penalty | $\lambda\sum\beta_j^2$ | $\lambda\sum\vert\beta_j\vert$ |
| Closed form｜闭式解 | ✅ | ❌ coordinate descent / LARS |
| Sparsity｜稀疏性 | ❌ shrinks toward 0, never to 0<br>趋近 0 但不为 0 | ✅ **exact zeros**｜**精确为 0**，自动特征选择 |
| $p > n$ | ✅ | ✅ but caps at $n$ features |
| Correlated group｜相关特征组 | Splits weight evenly (stable)<br>平均分配权重，稳定 | Picks one arbitrarily (unstable)<br>随机挑一个，不稳定 |
| Bayesian prior | Gaussian | Laplace |
| Best for｜适用场景 | All features matter; collinearity present<br>特征都有用 + 存在共线性 | Many features, most irrelevant<br>特征多且大部分无用 + 要可解释性 |
| Convexity | **Strictly** convex → unique solution<br>**严格凸**，解唯一 | Convex, not strictly<br>凸但非严格 |

### Three rules people break｜实践三铁律

1. **Standardize your features first.** The penalty treats all coefficients identically, so a feature in dollars and one in thousands of dollars get wildly unequal treatment.
   > **必须先标准化。** 惩罚项对所有系数一视同仁 —— 如果一个特征单位是「元」另一个是「万元」，惩罚力度就完全不公平。这是硬要求，不是建议。
2. **Don't penalize the intercept.** It only shifts the fit; penalizing it prevents the model from matching $\bar y$.
   > **不要惩罚截距。** 截距只负责平移，惩罚它会让模型无法拟合 $y$ 的均值，引入系统性偏差。sklearn 帮你处理了，**自己手写实现时会栽在这里**。
3. **Choose $\lambda$ by cross-validation.** Plot the regularization path（正则化路径）. A robust default is the **one-standard-error rule**.
   > **$\lambda$ 用交叉验证选**，并画出正则化路径看系数轨迹。$\lambda\to0$ 退化为 OLS，$\lambda\to\infty$ 全部归零。
   >
   > 更稳健的做法是 **one-standard-error rule**：在 CV error 落在最优值一个标准误以内的所有 $\lambda$ 中，**选最大的那个**（模型最简）。理由是 CV 曲线本身有噪声，最优点未必真的最优，往简单方向偏一点更稳。

> **📌 中文小结｜§8**
>
> - **共线性：估计仍无偏，但方差爆炸**；**基本不影响预测，只影响可解释性**。用 **VIF** 诊断（>5 警戒，>10 严重）。做预测可忽略，做归因必须处理。
> - **正则化的动机**：Gauss-Markov 只在「无偏」约束下最优，而我们从没真的需要无偏。**用一点 Bias 换大量 Variance，总 MSE 下降**。
> - **Ridge**：闭式解 $(X^TX+\lambda I)^{-1}X^Ty$，$p>n$ 也有解。**SVD 视角 —— 在低方差方向上收缩最狠，正是共线性所在的方向**（这条最加分）。
> - **Lasso**：无闭式解，正交设计下是 **soft-thresholding** —— 小于 $\lambda$ 的直接归零。
> - **L1 稀疏的三套解释**：几何（菱形尖角在坐标轴上）、次梯度（推力恒为 $\pm1$ vs. $2\beta\to0$）、贝叶斯（Laplace 先验在 0 处有尖峰）。**准备三套，讲两套。**
> - **完整对称**：`Loss ← 噪声分布`，`Regularizer ← 参数先验`。
> - 相关特征组 → **Elastic Net**（grouping effect）。
> - **三铁律**：先标准化、不罚截距、CV 选 $\lambda$（可用 one-SE rule）。

---

## 9. Writing it from scratch

Coding rounds like this one because it's short enough to finish and deep enough to probe.

> Coding 轮很喜欢这题：短到能写完，又深到能追问。下面这版覆盖了全部考点 —— 两种求解方式、Ridge、以及推断。

```python
import numpy as np


class LinearRegression:
    """OLS / Ridge via closed form or gradient descent."""

    def __init__(self, method="closed_form", lr=0.01, n_iters=1000, l2=0.0):
        self.method, self.lr, self.n_iters, self.l2 = method, lr, n_iters, l2

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Xb = self._add_bias(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, d = Xb.shape

        if self.method == "closed_form":
            A = Xb.T @ Xb
            if self.l2 > 0:
                I = np.eye(d)
                I[0, 0] = 0.0                    # 不惩罚截距 — never penalize the intercept
                A = A + self.l2 * I
            # 用 solve 而非 inv：更快，且数值稳定性好一个数量级
            self.w = np.linalg.solve(A, Xb.T @ y)
        else:
            self.w = np.zeros(d)
            self.history = []
            for _ in range(self.n_iters):
                resid = Xb @ self.w - y
                grad = Xb.T @ resid / n
                if self.l2 > 0:
                    reg = self.l2 * self.w
                    reg[0] = 0.0
                    grad += reg / n
                self.w -= self.lr * grad
                self.history.append(np.mean(resid ** 2))
        return self

    def predict(self, X):
        return self._add_bias(np.asarray(X, dtype=float)) @ self.w

    def r2(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        rss = np.sum((y - self.predict(X)) ** 2)
        tss = np.sum((y - y.mean()) ** 2)
        return 1 - rss / tss

    def summary(self, X, y):
        """标准误、t 统计量、95% 置信区间（需满足经典假设）"""
        Xb = self._add_bias(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, d = Xb.shape
        resid = y - Xb @ self.w
        sigma2 = resid @ resid / (n - d)         # 除自由度而非 n → 无偏估计
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xb.T @ Xb)))
        return {"coef": self.w, "se": se, "t": self.w / se,
                "ci_low": self.w - 1.96 * se, "ci_high": self.w + 1.96 * se}
```

And the two diagnostics from §7 and §8, which occasionally get asked as a standalone follow-up:

> 第 7、8 节的两个诊断量，偶尔会被单独拎出来当追加题。

```python
def vif(X):
    """方差膨胀因子 VIF_j = 1 / (1 - R_j^2)"""
    X = np.asarray(X, dtype=float)
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, X[:, j]).r2(others, X[:, j])
        out.append(np.inf if np.isclose(r2, 1.0) else 1.0 / (1.0 - r2))
    return np.array(out)


def cooks_distance(X, y):
    """D_i = r_i^2 / (d * sigma^2) * h_ii / (1 - h_ii)^2"""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    n, d = Xb.shape
    H = Xb @ np.linalg.solve(Xb.T @ Xb, Xb.T)    # hat matrix 帽子矩阵
    h = np.diag(H)
    resid = y - H @ y
    sigma2 = resid @ resid / (n - d)
    return resid ** 2 / (d * sigma2) * h / (1 - h) ** 2
```

**The details that actually matter here｜这段代码里真正要紧的细节**

| Question | Answer |
|---|---|
| Why not `np.linalg.inv`? | Explicit inversion is slower and numerically worse. `solve` does LU on the system directly; `lstsq` (SVD) for ill-conditioned problems.<br>**显式求逆又慢又不稳。`solve` 直接 LU 解方程组；病态问题用 `lstsq`（走 SVD）最安全。** |
| Why exclude the intercept from the penalty? | It only shifts the fit. Penalizing it stops the model from reaching $\bar y$.<br>**截距只负责平移，惩罚它会让模型够不到 $\bar y$，引入系统性偏差。** |
| Why $n-d$ and not $n$ in `sigma2`? | $RSS/n$ is the MLE but **biased low**. Dividing by df gives the unbiased estimate.<br>**$RSS/n$ 是 MLE 但低估了；除以自由度才无偏。** |
| GD isn't converging — what do you check? | Unscaled features, learning rate too large, or too few iterations. Watch `history`.<br>**① 特征没标准化（等高线太椭圆）；② 学习率过大（loss 发散）；③ 迭代不够。看 `history` 是否单调下降。** |
| Would this scale to 10M rows? | No — swap to mini-batch SGD and never materialize $X^TX$.<br>**不行 —— 换成 mini-batch SGD，且绝不物化 $X^TX$。** |

---

## 10. What actually needs to stick

**真正需要记住的，其实没有那么多**

Most of this post is reasoning you can reconstruct. A small part isn't — it's the handful of distinctions that, once you have them, generate the rest.

> 前面九节里的绝大部分内容，理解之后都能**当场推回来**。真正需要单独记住的只有一小撮 —— 那些一旦握住、其余就能自己长出来的区分。

### The rapid-fire round｜快问快答

按主题分组。每题下面的 🔑 标的是**最常被漏掉的那半句** —— 通常不是因为不会，而是因为答完前半句就停了。

**On assumptions｜假设**

- *What are the assumptions?* → Two tiers. Linearity, $\mathbb{E}[\epsilon\vert X]=0$, no perfect collinearity give **unbiasedness and consistency**. Add spherical errors for **Gauss-Markov / BLUE**. Normality is separate — **exact small-sample inference only**.
  > 🔑 **必须分两层说**，这是全场最高 ROI 的一句话。
- *Why doesn't normality affect unbiasedness?* → The proof only uses $\mathbb{E}[\epsilon\vert X]=0$. The error distribution never enters. In large samples, the CLT gives $\hat\beta$ an asymptotically normal **sampling distribution** without $\epsilon$ being normal.
  > 🔑 **CLT 保证的是 $\hat\beta$ 的抽样分布，不是 $\epsilon$ 的分布**。
- *What's BLUE, precisely?* → Minimum variance **within the linear-and-unbiased class**. Drop the unbiasedness constraint and Ridge can win on MSE.
  > 🔑 **BLUE 是受限比赛的冠军**，不是绝对冠军 —— 这句直通正则化。
- *Which violation is worst?* → **Endogeneity**, because it makes $\hat\beta$ biased *and* inconsistent. Everything else only breaks inference.
  > 🔑 **只有内生性毁估计，其余只毁推断**。

**On loss and estimation｜损失与估计**

- *Why squared loss?* → It's the Gaussian NLL; it's convex, smooth, and closed-form; and its minimizer is the **conditional mean**.
  > 🔑 三条都要说，**从条件均值那条开始说最有高度**。
- *Heavy-tailed noise?* → L1, targeting the **conditional median**; or **Huber** in practice.
  > 🔑 **换 loss = 换目标统计量**，这才是重点，不只是「更稳健」。
- *Can it fit nonlinear relationships?* → Yes, through feature transforms — it's linear in $\beta$, not in $x$.
  > 🔑 答「能」，然后补「但它不会自己发现该做什么变换」。
- *Closed form or gradient descent?* → Closed form when $p$ is small; SGD when $n$ or $p$ is large; SVD or Ridge when ill-conditioned. And never call `inv()`.
  > 🔑 别忘了最后半句 —— **`inv()` 是减分项**。
- *Do I need to scale features?* → **Not for OLS.** Yes for gradient descent, yes for any regularization, yes if comparing coefficient magnitudes.
  > 🔑 **纯 OLS 不用标准化** —— 多数人答错这个。

**On inference and diagnostics｜推断与诊断**

- *Is high $R^2$ good?* → Not by itself. It never decreases when you add features, it can go negative out-of-sample, and Anscombe's quartet shows identical $R^2$ across wildly different data.
  > 🔑 结尾一定要落到 **"plot the residuals"**。
- *Why $F$ if you have $t$?* → $F$ is a global test, immune to multiplicity. And **significant $F$ with no significant $t$ signals collinearity**.
  > 🔑 后半句是白送的加分。
- *What's a p-value?* → Probability of data this extreme **given $H_0$**. Not the probability $H_0$ is true, not a measure of effect size.
  > 🔑 **条件方向**是最常见的错误。
- *How do you validate assumptions?* → Residuals-vs-fitted, Q-Q, scale-location, residuals-vs-leverage with Cook's D. Plus residual ACF for time series.
  > 🔑 四张图 + 时序加 ACF。
- *Outlier vs. leverage vs. influence?* → Unusual in $y$ / unusual in $x$ / **changes the coefficients when removed**. Cook's D $\propto r_i^2 \cdot \frac{h_{ii}}{1-h_{ii}}$.
  > 🔑 **影响力 = 残差 × 杠杆的乘积**；发现影响点**不要说「删掉」**。

**On collinearity and regularization｜共线性与正则化**

- *Does collinearity hurt prediction?* → **Barely.** It hurts coefficient stability and interpretability. VIF > 5 warns, > 10 is serious.
  > 🔑 **「不影响预测」要先说** —— 这是最反直觉、最能体现你想清楚了的一点。
- *Ridge vs. Lasso?* → L2, closed form, shrinks-but-never-zeros. L1, no closed form, **exact zeros**. Correlated groups → Elastic Net.
  > 🔑 别忘了 Elastic Net 那句收尾。
- *Why is L1 sparse?* → Pick two: diamond corners on the axes; constant $\pm1$ subgradient versus a vanishing $2\beta$; soft-thresholding in the orthonormal case.
  > 🔑 **准备三套，讲两套**；次梯度那套最本质。
- *Why does Ridge fix collinearity?* → Adding $\lambda I$ lifts every eigenvalue, cutting the condition number. Via SVD, it shrinks **hardest along low-variance directions**.
  > 🔑 **SVD 那句是全文最值钱的加分点之一**。
- *Bayesian reading?* → MAP with a prior. **Gaussian prior → Ridge, Laplace prior → Lasso.**
  > 🔑 配合 `Loss ← 噪声，Regularizer ← 先验` 一起说，形成完整对称。
- *When would you still choose a linear model over GBDT?* → Interpretability under regulatory scrutiny; small $n$ with large $p$; genuine statistical inference; ultra-low latency; safer extrapolation; and always as a **baseline**.
  > 🔑 **「树模型完全无法外推」**是个很少有人提但很扎实的点。

### EN ↔ 中文 术语对照表

中文学的、英文答的，卡壳往往不是不会，而是**术语一时接不上**。

| English | 中文 | English | 中文 |
|---|---|---|---|
| Exogeneity | 外生性 | Endogeneity | 内生性 |
| Homoscedasticity | 同方差性 | Heteroscedasticity | 异方差性 |
| Autocorrelation | 自相关 | Spherical errors | 球形误差 |
| Unbiased / Consistent | 无偏 / 一致 | Efficiency | 有效性 |
| Sampling distribution | 抽样分布 | Degrees of freedom | 自由度 |
| Normal equation | 正规方程 | Hat matrix | 帽子矩阵 |
| Idempotent | 幂等 | Column space | 列空间 |
| Leverage | 杠杆值 | Influential point | 强影响点 |
| Goodness of fit | 拟合优度 | Residual | 残差 |
| Multicollinearity | 多重共线性 | Variance inflation factor | 方差膨胀因子 |
| Ill-conditioned | 病态的 | Condition number | 条件数 |
| Shrinkage | 收缩 | Soft-thresholding | 软阈值 |
| Regularization path | 正则化路径 | Grouping effect | 分组效应 |
| Prior / Posterior | 先验 / 后验 | MAP estimation | 最大后验估计 |
| Conditional mean / median | 条件均值 / 中位数 | Quantile | 分位数 |
| Misspecification | 模型设定错误 | Overfitting | 过拟合 |

### Reconstruct these and you have it｜能复原这些，就说明是真懂了

Not a list to memorize — a list to *derive*. Each item below is something you should be able to rebuild from scratch; if you can, everything else in this post follows from it.

> 这不是背诵清单，是**复原清单**。下面每一条都应该能从零推回来 —— 推得回来，前面九节的其余内容自然也在手里了。

- [ ] 把假设分成两层，说清每层各买到什么
- [ ] 推导 $\mathbb{E}[\hat\beta\vert X] = \beta$，并指出正态性**恰好没有出现**在哪一步
- [ ] 逐字母拆解 BLUE，把 "U" 连到正则化上
- [ ] 给出正规方程的**三种**推导：最小二乘、MLE、正交投影
- [ ] 给出平方损失的**三个**理由，落在条件均值上
- [ ] 应对重尾追问：L1 → 条件中位数 → 工程上用 Huber
- [ ] 任一假设违背，按 **后果 → 诊断 → 修复** 三段答
- [ ] 说出残差诊断四张图各自抓什么
- [ ] 区分 outlier / leverage / influence，把 Cook's D 读成乘积
- [ ] 说清共线性**毁什么、不毁什么**
- [ ] 用**两种**方式解释 L1 稀疏
- [ ] 讲出 Ridge 在低方差方向收缩最狠的 SVD 视角
- [ ] 说出完整对称：`Loss ← 噪声，Regularizer ← 先验`
- [ ] 从零写出 `LinearRegression` 类

---

## Closing thought

Everything above linear regression in the stack is, in a sense, the same four questions asked with the answers hidden: *what distribution am I assuming, what loss does that imply, what prior am I smuggling in as a penalty, and what breaks when the assumptions don't hold.* In a gradient-boosted forest you can't read those answers off anything. Here they're sitting in the equations.

So the thing worth taking away isn't the model. It's the habit. Once you've asked those four questions somewhere they have exact answers, you keep asking them in places where they don't — and that's where they're worth the most.

The five-assumption recital never gets you there. The two-tier structure does.

> **线性回归之上的所有模型，本质上都在回答同样的四个问题** —— 只是答案藏起来了：
>
> 我假设了什么分布？这个假设推出什么损失？我以惩罚项的名义偷偷塞进了什么先验？假设不成立时会坏在哪里？
>
> 在 GBDT 里，这四个问题的答案你从任何地方都读不出来。在这里，它们就明写在方程里。
>
> 所以真正该带走的不是这个模型，是**这个习惯** —— 当你在一个有精确答案的地方练过这四问，你会在没有精确答案的地方继续问。而后者才是它最值钱的时候。
>
> 背五条假设到不了那里。分两层可以。

---

*All figures generated with matplotlib; the [source scripts](https://github.com/yangnyc1024/yangnyc1024.github.io/tree/main/assets/posts/linear-regression-field-guide/src) are on GitHub. Reuse them freely.*

*If you found an error, or there's a corner of this I've glossed over, I'd genuinely like to hear about it.*
