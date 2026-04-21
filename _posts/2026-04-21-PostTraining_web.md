---
title: "LLM Post-Training, Explained"
subtitle: "大语言模型后训练全景"
date: 2026-04-21 12:00:00 +0800
categories:
  - LLM
  - Post-Training
tags:
  - llm
  - post-training
  - sft
  - dpo
  - ppo
  - grpo
description: "A bilingual overview of how post-training reshapes a base model into a usable assistant."
---

# LLM Post-Training, Explained
# 大语言模型后训练全景

**From Supervised Fine-Tuning to Reinforcement Learning / 从监督微调到强化学习**

A bilingual overview of how post-training reshapes a base model into a usable assistant.
一篇双语综述，说明后训练如何把基础模型塑造成可用的助手。

2026-04-21

**Overview / 提要**

- **SFT** learns from demonstrations. / **SFT** 从示范中学习。
- **DPO** learns from preferred-vs-rejected comparisons. / **DPO** 从“优选 vs 劣选”的比较中学习。
- **PPO** learns online from sampled responses, rewards, and a critic. / **PPO** 通过在线采样、奖励和 critic 学习。
- **GRPO** keeps online optimization but replaces the critic with group-relative comparison. / **GRPO** 保留在线优化，但用组内相对比较替代 critic。
- A useful summary is that post-training turns next-token prediction into a form of goal-directed sequential decision making. / 可以把后训练概括为：将 next-token prediction 转化为一种面向目标的序列决策过程。

**Reading Guide / 阅读说明**

The discussion proceeds from motivation to problem formulation, then to the objectives of SFT, DPO, PPO, and GRPO, and finally to practical method selection.

全文按照以下顺序展开：先说明为什么需要后训练，再讨论问题形式化，然后依次介绍 SFT、DPO、PPO、GRPO 的优化目标，最后讨论实际使用时的方法选择。

## Why Post-Training Matters / 为什么后训练重要

Pretraining gives a model fluency, broad knowledge, and the ability to continue text. What it does *not* automatically give is the product behavior we care about: following instructions, staying on format, respecting preferences, behaving safely, or reasoning reliably under constraints.

预训练给模型带来的是流畅性、广泛知识和续写能力；但它不会自动给出产品真正关心的行为：听懂指令、稳定遵守格式、符合人类偏好、足够安全，以及在约束下可靠推理。

Post-training is the phase where we reshape a base model into a policy that better matches human goals or task-defined goals. Depending on the available data, that goal may be imitation, pairwise preference alignment, online reward maximization, or success on verifiable tasks such as math, code, or tool use.

后训练就是把基础模型重新塑形成一个更符合人类目标或任务目标的策略。具体目标取决于你手里有什么数据：可能是模仿示范、对齐偏好、在线最大化奖励，或者提升数学、代码、工具调用这类可验证任务的成功率。

**Core intuition / 核心直觉**

If pretraining teaches a model to *speak*, post-training teaches it how to *behave*.

如果说预训练教会模型“会说话”，那么后训练教会模型“怎么做事”。

## A Unifying Lens: LLMs as Sequential Decision Makers / 统一视角：把 LLM 看成序列决策系统

A clean way to understand post-training is to reinterpret generation as a sequential decision problem. Given a prompt $x$, the model generates a response $y=(y_1, y_2, \dots, y_T)$. At step $t$, the current context $(x, y_{<t})$ is the *state*, the next token $y_t$ is the *action*, and the model distribution $\pi_\theta(y_t \mid x, y_{<t})$ is the *policy*, where $\theta$ denotes the trainable parameters.

理解后训练最清晰的方式，就是把生成过程重新看成一个序列决策问题。给定 prompt $x$，模型生成一条 response $y=(y_1, y_2, \dots, y_T)$。在第 $t$ 步，当前上下文 $(x, y_{<t})$ 就是*状态*，下一个 token $y_t$ 就是*动作*，而模型分布 $\pi_\theta(y_t \mid x, y_{<t})$ 就是*策略*；其中 $\theta$ 表示可训练参数。

Once a full answer is produced, we can assign a reward: helpfulness, correctness, safety, human preference, or a verifiable score from a checker. This is why RL words such as *policy*, *reward*, *return*, *value*, and *advantage* show up so naturally in modern LLM training.

当完整回答生成之后，我们就可以给出奖励：比如有帮助性、正确性、安全性、人类偏好，或者来自验证器的可验证分数。也正因为如此，*policy*、*reward*、*return*、*value*、*advantage* 这些 RL 词汇才会如此自然地出现在大模型训练里。

| Concept / 概念 | LLM interpretation / LLM 对应 | Why it matters / 为什么重要 |
| --- | --- | --- |
| State / 状态 | Prompt + partial response | Decides what information the model can condition on / 决定模型当前能看到什么信息 |
| Action / 动作 | Next token | The smallest unit of decision / 决策的最小单位 |
| Policy / 策略 | $\pi_\theta(y_t \mid x, y_{<t})$ | Controls how likely each continuation is / 决定每种续写的概率 |
| Reward / 奖励 | Preference, correctness, safety, verifier score | Tells the model what “better” means / 告诉模型什么叫“更好” |
| Value / 价值 | Expected future quality from the current prefix | Helps estimate how promising a partial answer is / 估计当前前缀未来还有多大潜力 |
| Trajectory / 轨迹 | A full generated response | The object that finally gets evaluated / 最终被评价的一整条回答 |

## One Prompt, Four Learning Signals / 一个问题，四种学习信号

Consider a simple prompt: `What is 12 + 7?` Even this small example is enough to distinguish the four main families of post-training methods.

看一个简单问题：`What is 12 + 7?`。即使是这样一个很小的例子，也足以区分后训练里最主要的四类方法。

**Same prompt, different supervision / 同一个问题，不同的监督方式**

- **SFT / 监督微调**: show the correct answer `12 + 7 = 19`. / 直接给标准答案 `12 + 7 = 19`。
- **DPO / 偏好优化**: show `19` as chosen and `18` as rejected. / 告诉模型 `19` 更好，`18` 更差。
- **PPO / 在线强化学习**: let the model answer first, then reward correct outputs and penalize wrong ones. / 先让模型自己回答，再对正确答案奖励、对错误答案惩罚。
- **GRPO / 组相对优化**: sample several answers such as `18`, `19`, `21`, `17`, then compare them within the group. / 一次生成多个候选，比如 `18`、`19`、`21`、`17`，再在组内做相对比较。

As we move from SFT to DPO to online RL, the signal becomes less static and more interactive. We move from *copying a target* to *ranking alternatives* to *learning from the model's own sampled behavior*.

从 SFT 到 DPO 再到在线 RL，监督信号会变得越来越“动态”。我们是在从*复制目标答案*，逐步走向*比较候选答案*，再走向*从模型自己采样出来的行为中学习*。

## SFT: The Foundation Layer / SFT：后训练的基础层

Supervised Fine-Tuning starts with prompt-response pairs and maximizes the probability of the target response. In implementation terms, it is usually just cross-entropy or negative log-likelihood over the response tokens.

监督微调从 prompt-response 配对数据出发，目标是最大化目标回答的概率。落到实现里，它通常就是 response token 上的交叉熵或负对数似然训练。

$$
\mathcal{L}_{\mathrm{SFT}}
=
-\sum_{i=1}^{N}\sum_{t=1}^{T_i}
\log p_\theta\!\big(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)}\big).
$$

In this objective, $\mathcal{L}_{\mathrm{SFT}}$ is the loss being minimized, $N$ is the number of prompt-response pairs, $T_i$ is the response length for example $i$, and the inner sum runs over response tokens. The same parameter vector $\theta$ is shared across all examples and all token positions.

在这个目标里，$\mathcal{L}_{\mathrm{SFT}}$ 表示要最小化的训练 loss，$N$ 是 prompt-response 对的数量，$T_i$ 是第 $i$ 个样本的 response 长度，内层求和沿着 response token 展开。同一个参数向量 $\theta$ 会在所有样本和所有 token 位置上共享。

This is why SFT is often compared to behavior cloning in RL. We do not ask the model to explore. We simply show what good behavior looks like and make it imitate.

这也是为什么 SFT 常被类比为 RL 里的 behavior cloning。它不要求模型探索，而是直接把“好的行为”展示给模型，然后让模型去模仿。

### Minimal code / 极简代码

```python
import torch
import torch.nn.functional as F

def sft_loss(logits, labels, ignore_index=-100):
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index,
    )
```

### Why SFT works well / 为什么 SFT 好用

- **Bootstrapping / 冷启动**: quickly teaches a new response style, output format, tool schema, or domain convention.
- **Stability / 稳定**: the objective is simple and well understood.
- **Scalability / 易扩展**: large amounts of curated prompt-response data are easy to pipeline.

### Where SFT falls short / SFT 的局限

- **No explicit ranking / 没有显式比较**: it does not tell the model *why* one answer is better than another.
- **No exploration / 不探索**: the model only learns from fixed demonstrations.
- **Data narrowness / 数据分布变窄**: if the data is too narrow, the model may overfit new patterns and regress elsewhere.

**Best use case / 最适合的场景**

Use SFT when you already know what the response should look like and can write it down clearly.

当你已经明确知道“理想回答长什么样”，而且可以直接把它写出来时，SFT 通常是第一选择。

## DPO: Learning from Comparisons / DPO：从比较中学习

Direct Preference Optimization sits between pure imitation and full online RL. Instead of asking the model to copy one target answer, DPO gives a prompt, a chosen response, and a rejected response, then optimizes the model to score the chosen one higher. ([rafailov2023dpo](#ref-rafailov2023dpo))

DPO 处在“纯模仿”和“完整在线 RL”之间。它不再只是让模型复制一个标准答案，而是给出同一个 prompt 下的 chosen response 与 rejected response，再优化模型，让它更偏向 chosen。

$$
\mathcal{L}_{\mathrm{DPO}}
=
-\mathbb{E}_{(x, y^{+}, y^{-}) \sim \mathcal{D}}
\left[
\log \sigma \Big(
\beta \big[
\log \frac{\pi_\theta(y^{+}\mid x)}{\pi_{\mathrm{ref}}(y^{+}\mid x)}
-
\log \frac{\pi_\theta(y^{-}\mid x)}{\pi_{\mathrm{ref}}(y^{-}\mid x)}
\big]
\Big)
\right].
$$

Here $\mathcal{L}_{\mathrm{DPO}}$ is again a loss to be minimized, $\mathcal{D}$ is a preference dataset, $y^{+}$ and $y^{-}$ denote the chosen and rejected responses for the same prompt $x$, $\pi_{\mathrm{ref}}$ is a frozen reference policy, $\beta$ controls how sharply the model separates the two responses, and $\sigma(\cdot)$ is the sigmoid function. Also, $\log \pi_\theta(y\mid x)$ means the log-probability of the whole response sequence $y$, i.e. the sum of token-level log-probabilities across the generated answer.

这里的 $\mathcal{L}_{\mathrm{DPO}}$ 同样表示要最小化的 loss，$\mathcal{D}$ 是偏好数据集，$y^{+}$ 和 $y^{-}$ 分别表示同一个 prompt $x$ 下的 chosen 与 rejected response，$\pi_{\mathrm{ref}}$ 是冻结的 reference policy，$\beta$ 控制模型把两个 response 拉开的力度，而 $\sigma(\cdot)$ 是 sigmoid 函数。另外，$\log \pi_\theta(y\mid x)$ 指的是整条 response 序列 $y$ 的对数概率，也就是回答中各个 token 对数概率之和。

The key idea is not “imitate this answer,” but “prefer this answer over that one.” Mathematically, DPO is best viewed as a classification-style loss over preference pairs rather than an on-policy RL objective. That makes DPO especially useful when the model is already roughly capable but keeps choosing the wrong tone, preference, stance, refusal pattern, or output style.

DPO 的关键不再是“模仿这个答案”，而是“让这个答案排在另一个答案前面”。从数学形式上看，DPO 更像是作用在偏好对上的一种 classification-style loss，而不是 on-policy RL 目标。因此，当模型已经大致有能力，但总是在语气、偏好、立场、拒答边界或输出风格上选错时，DPO 往往特别有效。

**Toy preference pair / 一个最小偏好对**

**Prompt / 问题**: Explain why regularization helps generalization.

**Chosen / 优选**: Regularization discourages overly complex models and reduces overfitting.

**Rejected / 劣选**: Regularization mainly makes training faster, so it generalizes better.

DPO teaches the model that the first answer should rank above the second.

DPO 教模型学到：第一个回答应该排在第二个前面。

### Minimal pseudocode / 极简伪代码

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp_pos, logp_neg, logp_ref_pos, logp_ref_neg, beta=0.1):
    logits = beta * ((logp_pos - logp_ref_pos) - (logp_neg - logp_ref_neg))
    return -F.logsigmoid(logits).mean()
```

This is not production code, but it captures the core structure: compare the preferred and rejected answers relative to a frozen reference model, then push the margin in favor of the chosen answer.

这不是生产级实现，但它已经抓住了 DPO 的核心结构：把 preferred 和 rejected 放到冻结参考模型的基线上比较，然后把这个差距往更有利于 chosen 的方向推。

### A practical rewrite example / 一个定向改写例子

A common DPO use case is identity or style rewriting. Suppose a model originally answers “I'm Qwen.” For the same identity prompt, we mark that original answer as rejected and a lightly edited answer such as “I'm Deep Qwen.” as chosen. DPO does not need to relearn all model behavior; it only needs to shift preference between competing responses.

一个常见的 DPO 用例，是做身份或风格改写。假设模型原本会回答 “I'm Qwen.”，那么对同一个身份类 prompt，我们可以把这个原回答标成 rejected，再把轻微改写后的 “I'm Deep Qwen.” 标成 chosen。DPO 并不需要把模型的全部行为重新学习一遍，而是在两个竞争回答之间调整偏好。

**Interpretation / 理解方式**

This example is best understood as local behavioral correction. In practice, tools such as `DPOTrainer`, `DPOConfig`, and the hyperparameter $\beta$ mainly control how strongly the model separates preferred and rejected responses.

这个例子更适合被理解为一种局部行为纠偏。在实践里，`DPOTrainer`、`DPOConfig` 以及超参数 $\beta$，主要控制的是模型把 preferred 和 rejected 拉开的力度。

### Why DPO is appealing / 为什么 DPO 有吸引力

- **Preference-aware / 显式偏好**: it directly encodes relative quality.
- **No separate reward model / 不必单独训练奖励模型**: the optimization is written directly in terms of log-probability differences.
- **Operationally simpler than online RL / 比在线 RL 更轻**: no on-policy sampling loop is required during training.

### Its limitations / 它的限制

- **Still offline / 仍然是离线**: the model does not learn from its current sampled behavior.
- **Depends on comparison quality / 依赖偏好数据质量**: noisy chosen-vs-rejected pairs can teach the wrong boundary.

## PPO: Online RL with a Critic / PPO：带 Critic 的在线强化学习

The big conceptual jump from SFT and DPO to PPO is that training becomes *online*. The model is no longer updated only from a frozen dataset; it generates fresh responses under its current policy, those responses are scored, and the model is updated using that newly collected experience. ([schulman2017ppo](#ref-schulman2017ppo); [ouyang2022instructgpt](#ref-ouyang2022instructgpt))

从 SFT、DPO 到 PPO，最大的概念变化是训练变成了*在线*的。模型不再只从一份静态数据集里学习，而是在当前策略下生成新回答，对这些回答打分，再用新鲜采样到的经验继续更新自己。

**Typical PPO loop for LLMs / 大模型里的典型 PPO 流程**

- Sample prompts / 采样 prompts
- Generate responses with the current policy / 用当前策略生成回答
- Score them with a reward model or verifier / 用奖励模型或验证器打分
- Estimate value and advantage / 估计 value 和 advantage
- Update the policy with clipping and regularization / 用 clipping 与正则项更新策略

$$
\mathcal{J}^{\mathrm{CLIP}}_{\mathrm{PPO}}(\theta)
=
\mathbb{E}_t\left[
\min\Big(
r_t(\theta)\hat{A}_t,\;
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\Big)
\right],
\quad
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

Here $\mathcal{J}^{\mathrm{CLIP}}_{\mathrm{PPO}}$ is written as a surrogate objective to maximize; in implementation, one usually minimizes its negative. The expectation $\mathbb{E}_t[\cdot]$ is over sampled time steps, $s_t$ and $a_t$ denote the state and action at step $t$, $\theta_{\mathrm{old}}$ is the policy used to collect the current batch of trajectories, $\hat{A}_t$ is an estimated advantage, and $\epsilon$ is the clipping radius. The operator $\operatorname{clip}(\cdot)$ restricts the ratio to a small interval around 1. Crucially, $r_t(\theta)$ here is the new-to-old policy probability ratio, not the reward itself; reward information enters the update through $\hat{A}_t$.

这里把 $\mathcal{J}^{\mathrm{CLIP}}_{\mathrm{PPO}}$ 写成一个要最大化的 surrogate objective；落到实现里，通常会最小化它的相反数。期望 $\mathbb{E}_t[\cdot]$ 是对采样时间步求平均，$s_t$ 和 $a_t$ 表示第 $t$ 步的状态与动作，$\theta_{\mathrm{old}}$ 是用于收集当前这批轨迹的旧策略参数，$\hat{A}_t$ 是 advantage 的估计量，而 $\epsilon$ 是 clipping 半径。算子 $\operatorname{clip}(\cdot)$ 的作用，是把比例限制在 1 附近的一个小区间内。最需要提醒的是：这里的 $r_t(\theta)$ 虽然写成 $r$，但它表示的是新旧策略的概率比，而不是 reward；reward 信息是通过 $\hat{A}_t$ 进入更新的。

Conceptually, PPO is best read as a clipped *policy-improvement term*. In full PPO implementations, one usually minimizes the negative objective above while also including a value loss and sometimes an entropy bonus; in LLM fine-tuning, KL regularization to a reference policy is also common.

从概念上看，PPO 最好理解成一个带 clipping 的*策略改进项*。在完整 PPO 实现里，通常会最小化上面目标的相反数，同时再加上 value loss，有时还会配 entropy bonus；而在大模型微调里，也常常会再加入相对于 reference policy 的 KL 正则。

The clipping term is what makes PPO practical. It allows policy improvement while preventing each update from moving too far from the previous policy. In LLM systems, a reference model or KL penalty is also commonly used to stop the model from drifting too aggressively.

clipping 项是 PPO 非常实用的原因。它允许策略变好，但又限制每次更新不要离旧策略太远。在大模型系统里，通常还会再配合参考模型或 KL 惩罚，避免模型发生过度漂移。

### A short intuition example / 一个短例子

Consider the prompt “Write a concise answer: what is the capital of France?” One sample says “Paris” and another says “Berlin.” PPO does not only learn that one final answer is right and the other is wrong. It asks a finer question: which token decisions along the way pushed the response toward a better outcome? That is exactly where token-level advantage becomes useful.

看一个简单 prompt：“Write a concise answer: what is the capital of France?”。一次采样回答成了 “Paris”，另一次却回答成了 “Berlin”。PPO 不只是学习“最终哪个答案对、哪个答案错”，它还会继续追问：生成路径上到底是哪些 token 决策，把回答一步步带向了更好的结果？这正是 token-level advantage 发挥作用的地方。

### Why value matters / 为什么 value 很重要

Rewards for language tasks are often sparse and noisy. A value model estimates the expected future return from a partial prefix, which helps reduce variance and gives finer credit assignment across the generated tokens.

语言任务中的奖励通常既稀疏又有噪声。value model 的作用，是从部分前缀出发估计未来期望回报，从而降低方差，并把“这次更新到底应该归功于哪些 token”这件事分得更细。

### Minimal pseudocode / 极简伪代码

```python
import torch

def simple_advantage(reward, value):
    return reward - value

def ppo_policy_loss(logprob_new, logprob_old, advantage, eps=0.2):
    ratio = torch.exp(logprob_new - logprob_old)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
    return -torch.min(unclipped, clipped).mean()
```

This is a one-step simplification. Real LLM PPO usually estimates advantage over many tokens and often relies on GAE or related techniques, but the core logic is unchanged: compare the new policy against the old one, scale updates by advantage, and clip the update so the policy does not move too aggressively. ([schulman2015gae](#ref-schulman2015gae); [schulman2017ppo](#ref-schulman2017ppo))

这是一种单步化的简化写法。真实的大模型 PPO 往往会在多 token 序列上估计 advantage，并常常借助 GAE 或相关技巧；但核心逻辑不变：比较新旧策略，用 advantage 调整更新方向，再通过 clip 防止策略更新得过猛。

### Two common reward styles / 两类常见奖励

- **Learned reward / 学习到的奖励**: a reward model trained from human preference data.
- **Verifiable reward / 可验证奖励**: exact answer matching, unit tests, tool success, structured constraints, or other deterministic checkers.

### Trade-offs / 取舍

- **Strong but expensive / 强但昂贵**: PPO can be powerful, but it needs more infrastructure and more memory.
- **Sensitive to reward quality / 对奖励质量敏感**: a weak reward signal can train the wrong behavior very efficiently.

## GRPO: Critic-Free Relative Optimization / GRPO：去掉 Critic 的相对优化

GRPO remains an online RL method, but removes the separate value model. Instead of using a critic to estimate future return, it samples a group of responses for the same prompt and computes *relative* advantages inside the group. ([shao2024deepseekmath](#ref-shao2024deepseekmath))

GRPO 仍然属于在线 RL 方法，但去掉了单独的 value model。它不再让 critic 去估计未来回报，而是对同一个 prompt 一次采样一组回答，然后在组内计算*相对* advantage。

$$
\mathcal{J}_{\mathrm{GRPO}}^{\mathrm{simple}}(\theta)
=
\mathbb{E}_{x,\{y_i\}_{i=1}^{G}}
\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\min\Big(
\rho_{i,t}(\theta)\hat{A}_{i,t},\;
\operatorname{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}
\Big)
\right].
$$

$$
\rho_{i,t}(\theta)
=
\frac{\pi_\theta(y_{i,t}\mid x, y_{i,<t})}
{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x, y_{i,<t})},
\qquad
\hat{A}_{i,t}
=
\tilde r_i
=
\frac{r_i - \bar{r}}{\operatorname{std}(r_1,\dots,r_G) + \varepsilon}.
$$

This is the cleanest outcome-supervision sketch: the same normalized group-relative reward is assigned to every token of response $y_i$. In the original DeepSeekMath presentation, GRPO is still a PPO-style clipped objective, and many implementations also add explicit KL regularization to a reference policy. ([shao2024deepseekmath](#ref-shao2024deepseekmath))

这是最简洁的 outcome-supervision 写法：同一个组内归一化奖励，会被广播到 response $y_i$ 的每个 token 上。在 DeepSeekMath 的原始表述里，GRPO 仍然是 PPO 风格的 clipped objective，而且很多实现还会显式加入相对于 reference policy 的 KL 正则。

Here $\mathcal{J}^{\mathrm{simple}}_{\mathrm{GRPO}}$ is again written as an objective to maximize. The expectation is over prompts and groups of sampled responses, $G$ is the group size, $y_i$ is the $i$-th sampled response, $|y_i|$ is its length, and $y_{i,t}$ is its $t$-th token. The scalar $r_i$ is the reward assigned to the whole response $y_i$, $\bar r$ is the mean reward within the group, $\operatorname{std}(r_1,\dots,r_G)$ is the within-group standard deviation, $\epsilon$ is the PPO-style clipping radius, and $\varepsilon$ is a small constant for numerical stability. In this simplified presentation, every token in response $y_i$ shares the same response-level signal $\tilde r_i$. In other words, GRPO uses $\rho_{i,t}$ for the policy ratio and reserves $r_i$ for reward.

这里的 $\mathcal{J}^{\mathrm{simple}}_{\mathrm{GRPO}}$ 同样写成一个要最大化的 objective。期望是对 prompt 以及对应的一组采样 response 求平均，$G$ 是组大小，$y_i$ 表示第 $i$ 个采样 response，$|y_i|$ 表示它的长度，而 $y_{i,t}$ 表示其中第 $t$ 个 token。标量 $r_i$ 是赋给整条 response $y_i$ 的奖励，$\bar r$ 是组内平均奖励，$\operatorname{std}(r_1,\dots,r_G)$ 是组内标准差，$\epsilon$ 是 PPO 风格的 clipping 半径，而 $\varepsilon$ 是数值稳定用的小常数。在这个简化写法里，response $y_i$ 中的每个 token 共享同一个 response-level 信号 $\tilde r_i$。也就是说，在 GRPO 里，策略概率比写成 $\rho_{i,t}$，而 $r_i$ 专门留给 reward。

A useful intuition is the following: if one sampled answer is better than the others in the same group, increase its probability; if it is worse, decrease it. The optimization is still token-wise, but the advantage now comes from group-relative normalized rewards rather than a learned critic.

一个方便理解的直觉是：如果某个采样回答在同组里比其他回答更好，就提高它的概率；如果更差，就降低它的概率。优化本身仍然是 token-wise 的，但 advantage 不再来自 learned critic，而是来自组内归一化后的相对奖励。

**Practical Features of GRPO / GRPO 的实践特点**

- **No value model / 不需要 value model**: lower memory footprint and less system complexity.
- **Natural for verifiable tasks / 很适合可验证任务**: math, code, structured reasoning, and tool-driven tasks benefit from clear relative scoring.
- **Good when critic training is unstable / 当 critic 训练不稳定时更有吸引力**: relative comparison can be simpler than learning a strong value baseline.

### A math example / 一个数学例子

**One prompt, four candidates / 一个问题，四个候选**

**Prompt / 问题**: `A store sells 3 apples for $6. How much do 5 apples cost at the same rate?`

1. `$10`
2. `$12`
3. `$8`
4. `$9`

If the verifier says `$10` is correct, GRPO compares the four candidates relative to one another and pushes the model toward answers that look more like the winner.

如果验证器判断 `$10` 正确，GRPO 就会在这四个候选之间做相对比较，并把模型往更像“赢家”的回答分布上推。

### Minimal sketch / 极简示意

```python
import torch

def relative_advantages(rewards):
    mean = rewards.mean()
    std = rewards.std(unbiased=False).clamp_min(1e-6)
    return (rewards - mean) / std

rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
advantages = relative_advantages(rewards)
print(advantages)
```

The point is not the exact code shape, but the learning signal: GRPO often replaces a separately trained critic with a normalized within-group comparison.

这里要抓住的重点不是代码外形本身，而是学习信号：GRPO 常常用组内归一化后的相对比较，来替代单独训练一个 critic。

In practice, GRPO is particularly suitable for reasoning datasets with verifiable rewards. A checker can inspect the final `\boxed{}` answer in math tasks, assign reward automatically, and make within-group comparison easier to define and reproduce.

在工程实践里，GRPO 特别适合带可验证奖励的推理数据集。比如在数学任务里，验证器可以直接检查最终的 `\boxed{}` 答案，对正确性自动打分，这会让组内比较更容易定义，也更容易复现。

### What to watch out for / 需要注意什么

- **Relative does not mean perfect / 相对比较不代表绝对正确**: the “best in the group” can still be bad.
- **Reward quality still dominates / 奖励质量仍然决定上限**: if the verifier is weak or gameable, the optimization target is still flawed.

## Offline vs Online: Why the Distinction Matters / Offline 与 Online：为什么这个区分很重要

SFT and DPO are *offline* methods because they learn from a fixed dataset. PPO and GRPO are *online* methods because they generate fresh responses during training and update from those newly generated samples. This distinction matters because offline methods are usually simpler and more stable, while online methods are closer to true policy optimization.

SFT 和 DPO 属于 *offline* 方法，因为它们是从固定数据集学习；PPO 和 GRPO 属于 *online* 方法，因为它们会在训练过程中动态生成新回答，并基于这些新样本继续更新。这个区分很重要，因为离线方法通常更简单、更稳定，而在线方法则更接近真正的策略优化。

One useful intuition---not a universal law---is that online RL can sometimes be less distribution-shifting because it updates the model on samples drawn from its current policy. By contrast, SFT may pull the model toward target outputs that are farther from its current distribution. Whether this preserves capability better depends heavily on reward design, data coverage, and evaluation.

一个有帮助但并非放之四海而皆准的直觉是：online RL 有时会更少造成 distribution shift，因为它是在模型当前策略实际采样出来的响应上继续更新；而 SFT 则可能把模型往离当前分布更远的目标答案上拉。它是否真的更能保持能力，仍然高度依赖 reward 设计、数据覆盖和评测方式。

**Summary / 小结**

Offline methods optimize against a fixed snapshot of behavior; online methods optimize against the model's current behavior.

离线方法是在固定行为快照上优化；在线方法是在模型当前行为上持续优化。

## Choosing the Right Tool / 怎么选择合适的方法

| Method / 方法 | Training data / 数据形式 | Main signal / 主要信号 | Online? / 是否在线 | Best fit / 最适合 |
| --- | --- | --- | --- | --- |
| SFT | prompt + response | imitation / 模仿 | No / 否 | bootstrapping, format control, instruction following / 冷启动、格式控制、指令跟随 |
| DPO | prompt + chosen + rejected | preference comparison / 偏好比较 | No / 否 | behavior correction, alignment, preference shaping / 行为纠偏、对齐、偏好塑形 |
| PPO | prompt + sampled responses + reward + value estimate | scalar reward + token-level advantage / 标量奖励 + token 级 advantage | Yes / 是 | strongest but costly online optimization / 强但昂贵的在线优化 |
| GRPO | prompt + grouped sampled responses | group-normalized reward $\rightarrow$ token-wise advantage / 组内归一化奖励 $\rightarrow$ token 级 advantage | Yes / 是 | verifiable reasoning and lower-memory online RL / 可验证推理、低显存在线 RL |

**A Common Selection Order / 常见选择顺序**

- Start with **SFT** if you have good demonstrations. / 如果有高质量示范数据，先从 **SFT** 开始。
- Move to **DPO** when the issue is not basic capability but preference or response style. / 如果问题不在“不会做”，而在“选错风格或偏好”，就上 **DPO**。
- Use **PPO** or **GRPO** only when reward is trustworthy and the infrastructure is ready. / 只有当奖励足够可靠、基础设施也到位时，再考虑 **PPO** 或 **GRPO**。

## Why RL Vocabulary Is Useful / 为什么 RL 词汇有用

The RL terms used in LLM post-training are not merely stylistic borrowings. *Return* is the total reward of a sampled response. *Value* estimates expected future return from a prefix. *Advantage* measures how much better a sampled action was than a baseline expectation. ([sutton2018rl](#ref-sutton2018rl))

大模型后训练里使用的 RL 术语，并不只是修辞性的借用。*return* 是一条采样回答的总回报；*value* 是从当前前缀出发的未来期望回报；*advantage* 则衡量某个采样动作到底比基线预期好多少。

This is also where the connection to dynamic programming becomes clearer. PPO carries the flavor of approximate dynamic programming because it learns a value function to support policy improvement. GRPO removes that value function, so it feels more Monte-Carlo-like and less like explicit value-based estimation. This is best read as intuition, not as an exact equivalence between modern LLM training and classical tabular RL. ([sutton2018rl](#ref-sutton2018rl); [schulman2017ppo](#ref-schulman2017ppo); [shao2024deepseekmath](#ref-shao2024deepseekmath))

这也是它和动态规划连接最清晰的地方。PPO 带有 approximate dynamic programming 的味道，因为它会学习 value function 来辅助策略改进；GRPO 则去掉了 value function，因此更像一种 Monte Carlo 风格的方法，而不是显式的 value-based 估计。但这里最好把它理解成一种直觉类比，而不是把现代大模型训练和经典表格 RL 当成精确等价物。

The clean classical RL template behind this discussion is the MDP tuple $(S, A, P, R, \gamma)$. LLM post-training does not literally solve a tabular MDP, but the mapping remains useful as a mental model: prompt plus prefix is the state, the next token is the action, appending a token is the transition, and response quality is the reward. ([sutton2018rl](#ref-sutton2018rl))

这背后的经典 RL 抽象模板，其实就是 MDP 五元组 $(S, A, P, R, \gamma)$。大模型后训练并不是在字面意义上求解一个表格型 MDP，但把它当成理解问题的 mental model 依然很有帮助：prompt 加前缀是状态，下一个 token 是动作，接上一个 token 就是状态转移，而整条回答的质量就是奖励。

$$
(S, A, P, R, \gamma).
$$

In the classical MDP tuple, $S$ is the state space, $A$ is the action space, $P$ is the transition kernel, $R$ is the reward function, and $\gamma$ is the discount factor. These symbols are introduced here only as a conceptual bridge; the later LLM formulas are not direct tabular MDP updates.

在经典的 MDP 五元组里，$S$ 是状态空间，$A$ 是动作空间，$P$ 是状态转移核，$R$ 是奖励函数，而 $\gamma$ 是折扣因子。这里引入这些符号主要是为了建立概念桥梁；后面的大模型公式并不是对表格型 MDP 的直接更新。

**MDP components mapped to LLM training / 把 MDP 部件映射到 LLM 训练**

| State / 状态 | prompt + partial response / prompt 加当前回答前缀 |
| --- | --- |
| Action / 动作 | next token / 下一个 token |
| Transition / 转移 | append one token and move to a new prefix / 追加一个 token 并进入新的前缀状态 |
| Reward / 奖励 | helpfulness, correctness, safety, verifier score / 有帮助性、正确性、安全性、验证器分数 |
| Discount / 折扣 | often omitted or absorbed into finite-horizon return definitions / 在有限长度生成里常被省略，或吸收到 return 的定义里 |

**Intuition Only / 仅作直觉类比**

**Q-learning intuition**: “If I choose this token, what is the best continuation I could get?”

**Q-learning 的直觉**: “如果我选这个 token，后面最好的续写会是什么？”

**SARSA intuition**: “If I choose this token, what continuation will I get under my current behavior?”

**SARSA 的直觉**: “如果我选这个 token，按我当前策略，后面大概率会续写成什么？”

These analogies are useful for thinking, even though modern LLM post-training does not literally run tabular Q-learning or SARSA.

这种类比有助于理解，但现代 LLM 后训练并不会真的去跑表格版的 Q-learning 或 SARSA。

### Q-learning and SARSA formulas / Q-learning 与 SARSA 的公式直觉

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[
r + \gamma \max_{a'} Q(s',a') - Q(s,a)
\Big].
$$

Q-learning imagines the *best continuation* available after the current action. In LLM language, the analogy is: if I choose this token now, what is the best continuation I could still reach from here? ([watkins1992qlearning](#ref-watkins1992qlearning); [sutton2018rl](#ref-sutton2018rl))

Q-learning 关心的是当前动作之后还能达到的 *最优续写*。翻译成大模型语言，就是：如果我现在选这个 token，接下来我理论上还能走到多好的续写？

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[
r + \gamma Q(s',a') - Q(s,a)
\Big].
$$

In both update rules, $\alpha$ is the learning rate, $\gamma$ is the discount factor, and $s'$ denotes the next state. The difference lies in the treatment of the next action: Q-learning uses $\max_{a'}Q(s',a')$, while SARSA uses the value of the action $a'$ actually chosen by the current behavior policy. ([watkins1992qlearning](#ref-watkins1992qlearning); [rummery1994online](#ref-rummery1994online); [sutton2018rl](#ref-sutton2018rl))

在这两个更新式里，$\alpha$ 是学习率，$\gamma$ 是折扣因子，$s'$ 表示下一状态。两者的差别主要在于对下一动作的处理：Q-learning 使用 $\max_{a'}Q(s',a')$，而 SARSA 使用当前行为策略实际选出的动作 $a'$ 所对应的价值。

SARSA instead follows the current behavior policy. In LLM language, that sounds like: if I choose this token now, what continuation will my current model actually keep generating, rather than the best imaginable one? ([rummery1994online](#ref-rummery1994online); [sutton2018rl](#ref-sutton2018rl))

SARSA 则跟随当前行为策略。放到大模型语境里，它更像是在问：如果我现在选这个 token，按我当前模型的真实行为，后面大概率会继续生成成什么样，而不是去想象理论上的最优续写。

**Connecting Classical RL to Post-Training / 把经典 RL 再连回后训练**

- **Q-learning / Q-learning**: think in terms of the best continuation. / 用“最优续写”来理解。
- **SARSA / SARSA**: think in terms of the continuation under the current behavior. / 用“当前策略下会发生的续写”来理解。
- **PPO / PPO**: think in terms of advantage-weighted continuation with a learned baseline. / 用“带基线的 advantage 加权续写”来理解。
- **GRPO / GRPO**: think in terms of relative continuation inside a sampled group. / 用“采样组内的相对续写比较”来理解。

## A Small End-to-End Example / 一个小型端到端例子

Consider a small arithmetic example. It is enough to show how the supervision signal becomes richer as we move from SFT to DPO and then to online RL.

考虑一个简单的四则运算例子。它已经足以说明：从 SFT 到 DPO，再到 online RL，监督信号是如何逐步变得更丰富的。

### SFT dataset / SFT 数据

```python
sft_data = [
    {"prompt": "2 + 3 =", "response": "5"},
    {"prompt": "7 + 8 =", "response": "15"},
    {"prompt": "9 - 4 =", "response": "5"},
]
```

### DPO dataset / DPO 数据

```python
dpo_data = [
    {
        "prompt": "2 + 3 =",
        "chosen": "5",
        "rejected": "6",
    },
    {
        "prompt": "7 + 8 =",
        "chosen": "15",
        "rejected": "14",
    },
]
```

### Online RL reward / 在线 RL 奖励

```python
def reward_fn(pred, ground_truth):
    return 1.0 if pred.strip() == ground_truth.strip() else 0.0
```

### GRPO-style generation / GRPO 风格采样

```python
candidates = ["14", "15", "16", "13"]
rewards = [reward_fn(c, "15") for c in candidates]
# -> [0, 1, 0, 0]
```

This example already makes the progression visible. SFT corresponds to matching a target answer; DPO corresponds to preferring one answer over another; online RL corresponds to sampling first and then updating from reward; GRPO corresponds to sampling several answers and updating from their relative comparison.

这个例子已经足以把整体路线说明清楚：SFT 对应于拟合目标答案，DPO 对应于在两个答案之间学习偏好，online RL 对应于先采样再根据奖励更新，而 GRPO 对应于多次采样后依据相对比较来更新。

## Engineering Considerations / 工程注意事项

- **Data quality beats algorithm cleverness / 数据质量往往比算法花样更重要**: a clean SFT or DPO dataset can outperform a fancier pipeline with noisy supervision.
- **Separate capability from preference / 把能力问题和偏好问题分开看**: first ask whether the model *cannot* do the task or whether it merely *chooses the wrong kind of answer*.
- **Prefer verifiable rewards first / 优先从可验证奖励入手**: online RL is far safer when reward comes from exact checks instead of vague human impressions alone.
- **Track regressions explicitly / 显式追踪能力回退**: if a model gets better on one task but worse elsewhere, the training pipeline still needs work.
- **Choose the simplest method that matches the signal / 用最简单但足够的方法**: do not jump to online RL if a well-curated SFT or DPO pipeline already solves the problem.

**Practical Guideline / 实践建议**

A common workflow is: **use SFT to establish basic behavior**, **use DPO to adjust preferences**, and introduce **PPO/GRPO only when reward is reliable and online optimization is necessary**.

一种常见的工作流是：**先用 SFT 建立基本行为**，**再用 DPO 调整偏好**，只有在**奖励可靠且确有必要进行在线优化**时，再引入 **PPO/GRPO**。

## Condensed Summary / 简要总结

A compact comparison is as follows: SFT improves behavior by imitation, DPO by pairwise comparison, PPO by online sampling with reward and a critic, and GRPO by online sampling with group-relative comparison instead of a critic. From the MDP viewpoint, all four methods improve the policy of a token-generating agent, but they differ in the learning signal and optimization structure they use.

可以把它们简要比较为：SFT 通过模仿改进行为，DPO 通过成对比较调整偏好，PPO 通过在线采样、奖励和 critic 更新策略，而 GRPO 则通过在线采样和组内相对比较来更新策略，但不单独使用 critic。从 MDP 视角看，这四种方法本质上都在优化一个逐 token 生成的策略，只是采用的学习信号和优化结构不同。

**A Slightly More Formal Comparison / 稍微正式一些的比较**

SFT is closest to behavior cloning, DPO is offline preference optimization with an implicit reward view, PPO looks like approximate policy improvement with value estimation, and GRPO is a critic-free relative policy optimization method that is especially attractive for verifiable reasoning tasks.

SFT 最接近 behavior cloning；DPO 可以看成带隐式 reward 视角的离线偏好优化；PPO 很像带 value estimation 的近似策略改进；GRPO 则是一种去 critic 的相对策略优化，尤其适合带可验证奖励的推理任务。

## Conclusion / 结语

A short concluding summary is this: SFT improves behavior through demonstration matching, DPO through preference comparison, PPO through online reward optimization with a critic, and GRPO through online optimization with group-relative comparison.

可以把全文的结论概括为：SFT 通过示范匹配改进行为，DPO 通过偏好比较调整输出，PPO 通过带 critic 的在线奖励优化更新策略，而 GRPO 则通过组内相对比较进行在线优化。

At a coarse level, SFT provides demonstrations, DPO provides pairwise judgments, PPO provides scalar rewards with a learned baseline, and GRPO provides group-relative reward comparisons.

粗略地说，SFT 提供的是示范，DPO 提供的是成对判断，PPO 提供的是带基线估计的标量奖励，而 GRPO 提供的是组内相对奖励比较。

A compact way to connect the whole discussion is: *LLM post-training turns next-token prediction into a form of goal-directed sequential decision making.*

如果用一句话概括全文，可以写成：*LLM 后训练，是把 next-token prediction 转化为一种面向目标的序列决策过程。*

## References / 参考文献

- <a id="ref-sutton2018rl"></a> **[sutton2018rl]** Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
- <a id="ref-watkins1992qlearning"></a> **[watkins1992qlearning]** Christopher J. C. H. Watkins and Peter Dayan, “Q-learning,” *Machine Learning*, 8(3--4):279--292, 1992.
- <a id="ref-rummery1994online"></a> **[rummery1994online]** G. A. Rummery and M. Niranjan, *On-line Q-Learning Using Connectionist Systems*, Technical Report CUED/F-INFENG/TR 166, University of Cambridge, 1994.
- <a id="ref-schulman2015gae"></a> **[schulman2015gae]** John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel, *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438, 2015.
- <a id="ref-schulman2017ppo"></a> **[schulman2017ppo]** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov, *Proximal Policy Optimization Algorithms*, arXiv:1707.06347, 2017.
- <a id="ref-ouyang2022instructgpt"></a> **[ouyang2022instructgpt]** Long Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*, arXiv:2203.02155, 2022.
- <a id="ref-rafailov2023dpo"></a> **[rafailov2023dpo]** Rafael Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS, 2023.
- <a id="ref-shao2024deepseekmath"></a> **[shao2024deepseekmath]** Zhihong Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, arXiv:2402.03300, 2024.
- <a id="ref-datawhaleeasyrl"></a> **[datawhaleeasyrl]** Datawhale, [EasyRL](https://datawhalechina.github.io/easy-rl/#/).
- <a id="ref-datawhaleposttraining"></a> **[datawhaleposttraining]** Datawhale, [post-training-of-llms](https://github.com/datawhalechina/post-training-of-llms).
