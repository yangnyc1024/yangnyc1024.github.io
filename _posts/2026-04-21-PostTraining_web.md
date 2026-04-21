---
title: "LLM Post-Training, Explained"
subtitle: "A practical map from SFT and DPO to PPO and GRPO."
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
description: "An overview of how post-training reshapes a base model into a usable assistant."
---

# LLM Post-Training, Explained
# 大语言模型后训练全景

**From Supervised Fine-Tuning to Reinforcement Learning / 从监督微调到强化学习**

A overview of how post-training reshapes a base model into a usable assistant.
一篇综述，说明后训练如何把基础模型塑造成可用的助手。

2026-04-21

**Overview / 提要**

- **SFT** learns from demonstrations. / **SFT** 从示范中学习。
- **DPO** learns from preferred-vs-rejected comparisons. / **DPO** 从“优选 vs 劣选”的比较中学习。
- **PPO** learns online from sampled responses, rewards, and value-function estimates. / **PPO** 通过在线采样、奖励和价值函数估计学习。
- **GRPO** keeps online optimization but replaces a separately trained critic with group-relative comparison. / **GRPO** 保留在线优化，但用组内相对比较替代单独训练的价值模型。
- Post-training turns next-token prediction into goal-directed sequential decision making. / 后训练把 next-token prediction 转化为面向目标的序列决策。

**Reading Guide**

The article begins with the role of post-training and a unified formulation, then discusses SFT and DPO, introduces only the RL concepts needed for PPO and GRPO, and closes with method selection plus engineering considerations.

## Why Post-Training Matters

Pretraining gives a model fluency, broad knowledge, and the ability to continue text. It does not, by itself, determine the behavior required in real use: instruction following, format control, preference alignment, safety, or reliable reasoning under constraints.

Post-training closes this gap. Starting from a base model, it uses demonstrations, preference comparisons, or reward signals to make model behavior match human objectives or task-defined objectives more closely.

**Core Point / 核心判断**

If pretraining teaches a model to *speak*, post-training teaches it how to *behave*.

如果说预训练教会模型“会说话”，那么后训练教会模型“怎么做事”。

## A Unifying Lens: LLMs as Sequential Decision Makers

Post-training can be formulated as a sequential decision problem. Given a prompt $x$, the model generates a response $y=(y_1, y_2, \dots, y_T)$. At step $t$, the context $(x, y_{<t})$ is the *state*, the next token $y_t$ is the *action*, and the model distribution $\pi_\theta(y_t \mid x, y_{<t})$ is the *policy*, where $\theta$ denotes the trainable parameters.

Once a complete response is produced, it can be assigned a reward: helpfulness, correctness, safety, human preference, or a verifier score. This is why terms such as *policy*, *reward*, *return*, *value*, and *advantage* are structurally relevant rather than merely rhetorical borrowings.

| Concept    | LLM interpretation                              | Why it matters                                      |
| ---------- | ----------------------------------------------- | --------------------------------------------------- |
| State      | Prompt + partial response                       | Decides what information the model can condition on |
| Action     | Next token                                      | The smallest unit of decision                       |
| Policy     | $\pi_\theta(y_t \mid x, y_{<t})$                | Controls how likely each continuation is            |
| Reward     | Preference, correctness, safety, verifier score | Tells the model what “better” means                 |
| Value      | Expected future quality from the current prefix | Helps estimate how promising a partial answer is    |
| Trajectory | A full generated response                       | The object that finally gets evaluated              |

## One Prompt, Four Learning Signals

Consider a simple prompt: `What is 12 + 7?` Even this small example is enough to distinguish the four main families of post-training methods.

**Same Prompt, Different Supervision**

- **SFT**: show the correct answer `12 + 7 = 19`.
- **DPO**: show `19` as chosen and `18` as rejected.
- **PPO**: let the model answer first, then reward correct outputs and penalize wrong ones.
- **GRPO**: sample several answers such as `18`, `19`, `21`, `17`, then compare them within the group.

As we move from SFT to DPO to online RL, the signal becomes less static and more interactive. We move from *copying a target* to *ranking alternatives* to *learning from the model's own sampled behavior*.

## SFT: The Foundation Layer

Supervised Fine-Tuning starts with prompt-response pairs and maximizes the probability of the target response. In implementation terms, it is usually just cross-entropy or negative log-likelihood over the response tokens.

$$
\mathcal{L}_{\mathrm{SFT}}
=
-\sum_{i=1}^{N}\sum_{t=1}^{T_i}
\log p_\theta\!\big(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)}\big).
$$

In this objective, $\mathcal{L}_{\mathrm{SFT}}$ is the loss being minimized, $N$ is the number of prompt-response pairs, $T_i$ is the response length for example $i$, and the inner sum runs over response tokens. The same parameter vector $\theta$ is shared across all examples and all token positions.

This is why SFT is often compared to behavior cloning in RL. We do not ask the model to explore. We simply show what good behavior looks like and make it imitate.

### Minimal Code

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

### Strengths of SFT

- **Effective initialization**: quickly establishes a new response style, output format, tool schema, or domain convention.
- **Stability**: the objective is simple and well understood.
- **Scalability**: curated demonstration data is straightforward to pipeline at scale.

### Limitations of SFT

- **No explicit ranking**: it does not tell the model *why* one answer is better than another.
- **No exploration**: the model only learns from fixed demonstrations.
- **Data narrowness**: if the data is too narrow, the model may overfit new patterns and regress elsewhere.

### Constructing SFT Data

In practice, SFT quality depends as much on data construction as on the loss itself. Three common routes are distillation, best-of-$K$ / rejection sampling, and filtering. Distillation uses a stronger model to generate responses for a weaker model; best-of-$K$ keeps the strongest candidate among multiple generations; filtering removes low-quality or low-diversity prompt-response pairs before training.

### Full Fine-Tuning or LoRA

SFT can be implemented either as full fine-tuning or parameter-efficient fine-tuning. Full fine-tuning updates all model weights and usually gives the largest capacity to learn new behavior, but it is the most expensive. LoRA-style adaptation adds small low-rank trainable modules on top of a frozen base model, which is cheaper and often good enough when the goal is format control, style transfer, or moderate instruction tuning.

**Best Use Case**

Use SFT when you already know what the response should look like and can write it down clearly.

## DPO: Learning from Comparisons

DPO lies between supervised imitation and online reinforcement learning. It does not ask the model to copy a single target response. Instead, it trains on pairwise preferences: for the same prompt, the chosen response should receive a higher score than the rejected one. ([rafailov2023dpo](#ref-rafailov2023dpo))

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

The loss compares the chosen and rejected responses relative to a frozen reference policy. Here $\beta$ controls the sharpness of the preference margin, and $\log \pi_\theta(y\mid x)$ denotes the log-probability of the whole response sequence, i.e. the sum of token-level log-probabilities across the generated answer.

The key question in DPO is not how to reproduce one answer, but how to order two competing answers. For this reason, DPO is especially effective when the model already has the required capability but repeatedly chooses the wrong tone, stance, refusal boundary, or output style.

**Preference Pair Example**

**Prompt**: Explain why regularization helps generalization.

**Chosen**: Regularization discourages overly complex models and reduces overfitting.

**Rejected**: Regularization mainly makes training faster, so it generalizes better.

DPO teaches the model that the first answer should rank above the second.

### Minimal Pseudocode

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp_pos, logp_neg, logp_ref_pos, logp_ref_neg, beta=0.1):
    logits = beta * ((logp_pos - logp_ref_pos) - (logp_neg - logp_ref_neg))
    return -F.logsigmoid(logits).mean()
```

This is not production code, but it captures the core structure: compare the preferred and rejected answers relative to a frozen reference model, then push the margin in favor of the chosen answer.

### A Targeted Editing Example

DPO is often used for localized identity or style editing. Suppose a model originally answers “I'm Qwen.” For the same identity prompt, that answer can be marked as rejected, while a lightly edited answer such as “I'm Deep Qwen.” is marked as chosen. The task is not to relearn the entire response distribution, but to change the ordering between two nearby answers.

**Interpretation**

This kind of example illustrates local preference correction: the model does not need to relearn everything it can say; it only needs to separate two nearby answers more clearly. The hyperparameter $\beta$ controls how strongly that separation is enforced.

### Where Preference Pairs Come From

Two construction patterns dominate DPO practice. The first is *correction*: start from the model's current answer, treat it as rejected, and edit it into a better chosen answer. The second is *best-vs-worst comparison*: sample multiple responses for the same prompt, then keep the best one as chosen and the worst one as rejected.

This local ordering is precisely why DPO is effective for targeted behavior editing. It does not require a complete specification of the ideal response distribution; it only requires a reliable ordering between competing answers. Pair construction, however, must avoid superficial cues. If chosen answers always contain the same accidental phrase or formatting marker, the model may learn the cue rather than the intended preference boundary.

### Strengths of DPO

- **Preference-aware**: it directly encodes relative quality.
- **No separate reward model**: it optimizes log-probability differences directly.
- **Operationally simpler than online RL**: no on-policy sampling loop is required during training.

### Limitations of DPO

- **Still offline**: the model does not learn from its current sampled behavior.
- **Depends on comparison quality**: noisy chosen-vs-rejected pairs can teach the wrong boundary.

## RL Concepts for Post-Training

Once post-training becomes online, a small set of RL concepts becomes necessary. The issue is not that LLM training literally runs a tabular solver, but that terms such as *return*, *value function*, *Bellman backup*, *model-free learning*, and *bootstrapping* describe the signal structure used by PPO and GRPO. ([sutton2018rl](#ref-sutton2018rl))

### MDP Formulation and Bellman Perspective

The clean classical template is the MDP tuple $(S, A, P, R, \gamma)$. In LLM post-training, the state is prompt plus prefix, the action is the next token, the transition is appending one token, and the reward is the quality of the final response. In finite-horizon generation, $\gamma$ is often implicit or absorbed into the return definition. ([sutton2018rl](#ref-sutton2018rl))

$$
V(s)=R(s)+\gamma \sum_{s'} P(s' \mid s)V(s').
$$

The Bellman equation makes one point clear: value is not only immediate reward, but discounted expected future value. This is why the value function in PPO can be read as an estimate of how much quality a partial response may still realize. ([sutton2018rl](#ref-sutton2018rl))

**MDP Components Mapped to LLM Training**

| Concept    | LLM interpretation                               |
| ---------- | ------------------------------------------------ |
| State      | prompt + partial response                        |
| Action     | next token                                       |
| Transition | append one token and move to a new prefix        |
| Reward     | helpfulness, correctness, safety, verifier score |
| Return     | total reward along a sampled answer              |
| Advantage  | observed outcome minus baseline expectation      |

### Model-based and Model-free Perspectives

If the transition kernel and reward function are known, classical dynamic programming can evaluate or improve a policy without collecting new experience. That is the model-based setting. Most LLM post-training is instead model-free: there is no exact environment model for text quality, so the model must sample responses, receive scores, and learn from those trajectories. ([sutton2018rl](#ref-sutton2018rl))

**Why the Distinction Matters**

- **SFT / SFT**: supervised imitation, not online RL.
- **DPO / DPO**: offline preference optimization, still not model-based control.
- **PPO and GRPO**: model-free online optimization from sampled behavior plus reward.

### Sampling, Bootstrapping, and Credit Assignment

Another key distinction is between *sampling* and *bootstrapping*. Monte Carlo methods wait for a complete sampled trajectory and use the realized return. Temporal-difference methods update earlier, combining sampled reward with an estimate of what follows. Dynamic programming also bootstraps, but it assumes the environment model is known. ([sutton2018rl](#ref-sutton2018rl); [schulman2015gae](#ref-schulman2015gae))

**Three Ways to Estimate Future Quality**

- **DP**: no sampling, yes bootstrapping, requires a known model.
- **Monte Carlo**: sampled complete returns, no bootstrapping.
- **TD**: sampled rewards plus bootstrapped next-value estimates.

This distinction places PPO and GRPO more clearly. PPO is closer to TD-style learning because it combines sampled responses with value-function estimates, often through advantage estimation. GRPO is closer to Monte Carlo-style learning because it compares complete sampled responses directly and does not train a separate value model. ([sutton2018rl](#ref-sutton2018rl); [schulman2017ppo](#ref-schulman2017ppo); [shao2024deepseekmath](#ref-shao2024deepseekmath))

### Q-learning and SARSA as References

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[
r + \gamma \max_{a'} Q(s',a') - Q(s,a)
\Big].
$$

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[
r + \gamma Q(s',a') - Q(s,a)
\Big].
$$

In both updates, $\alpha$ is the learning rate and $\gamma$ the discount factor. The difference lies in how the next step is treated: Q-learning uses the best next action through $\max_{a'}Q(s',a')$, whereas SARSA uses the next action actually taken by the current policy. In LLM language, this roughly mirrors the difference between the best conceivable continuation and the continuation the current model would in fact produce. ([watkins1992qlearning](#ref-watkins1992qlearning); [rummery1994online](#ref-rummery1994online); [sutton2018rl](#ref-sutton2018rl))

**Four Reference Intuitions**

- **Q-learning intuition**: best continuation after this token.
- **SARSA intuition**: continuation under the current behavior.
- **PPO intuition**: advantage-weighted continuation with a learned baseline.
- **GRPO intuition**: group-relative comparison among sampled continuations.

## PPO: Online Reinforcement Learning with a Value Function

The central shift from SFT and DPO to PPO is that training becomes *online*. The model is no longer optimized only against a fixed dataset; it must generate responses under its current policy, receive reward signals, and update from newly collected trajectories. ([schulman2017ppo](#ref-schulman2017ppo); [ouyang2022instructgpt](#ref-ouyang2022instructgpt))

**Typical PPO Loop for LLMs**

- Sample prompts
- Generate responses with the current policy
- Score them with a reward model or verifier
- Estimate value and advantage
- Update the policy with clipping and regularization

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

This is a clipped policy objective. In practice one usually minimizes its negative. The critical point is that $r_t(\theta)$ is the probability ratio between the new and old policies, not the reward itself; reward enters the update through the estimated advantage $\hat{A}_t$.

Conceptually, PPO can be read as a conservative policy-improvement step. Clipping prevents each update from moving too far away from the behavior policy used to collect the data. Full implementations usually add a value loss and, in LLM settings, often a KL penalty to a reference policy.

### A Short Example

Consider the prompt “Write a concise answer: what is the capital of France?” One sample says “Paris” and another says “Berlin.” PPO does not only learn that one final answer is right and the other is wrong. It asks a finer question: which token decisions along the way pushed the response toward a better outcome? That is exactly where token-level advantage becomes useful.

### The Role of the Value Function

Rewards in language tasks are often sparse and noisy. A value function estimates the expected future return of a partial prefix, which reduces variance and supports finer credit assignment across generated tokens.

### Minimal Pseudocode

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

### Two Common Reward Styles

- **Learned reward**: a reward model trained from human preference data.
- **Verifiable reward**: exact answer matching, unit tests, tool success, structured constraints, or other deterministic checkers.

Learned rewards are usually the right tool when the target is open-ended behavior such as helpfulness, harmlessness, tone, or refusal style, because these qualities are hard to define with exact rules. Verifiable rewards are usually stronger when the task has an unambiguous check, such as a math final answer, a code unit test, a SQL execution result, or a tool call that either succeeds or fails.

In practice, reward design includes not only the scoring rule but also the extraction pipeline that turns a raw response into something scorable. If a math reward checks the final `\boxed{}` answer or a code reward runs unit tests, that post-processing logic becomes part of the training objective itself.

### Costs and Constraints

- **Strong but expensive**: PPO can be powerful, but it needs more infrastructure and more memory.
- **Sensitive to reward quality**: a weak reward signal can train the wrong behavior very efficiently.

## GRPO: Relative Optimization Without a Separate Value Model

GRPO remains an online RL method, but it dispenses with a separately trained value model. Instead of estimating future return with a critic, it samples several responses for the same prompt and computes *relative* advantages within the group. ([shao2024deepseekmath](#ref-shao2024deepseekmath))

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

This simplified form assigns the same normalized group-relative signal to every token of response $y_i$. In the original DeepSeekMath formulation, GRPO is still optimized with a PPO-style clipped objective, and many implementations also add explicit KL regularization to a reference policy. ([shao2024deepseekmath](#ref-shao2024deepseekmath))

The notation emphasizes only three things: $G$ is the group size, $r_i$ is the response-level reward, and $\tilde r_i$ is the normalized relative signal derived from that reward. In the simplified form shown here, every token in response $y_i$ shares the same response-level signal, while $\rho_{i,t}$ plays the same ratio role as in PPO.

GRPO increases the probability of answers that rank better within the sampled group and suppresses answers that rank worse. The update is still token-wise, but the advantage now comes from normalized within-group comparison rather than a separately learned critic.

**Main Characteristics of GRPO**

- **No separate value model**: lower memory footprint and less system complexity.
- **Natural for verifiable tasks**: math, code, structured reasoning, and tool-driven tasks benefit from clear relative scoring.
- **Good when value-model training is unstable**: relative comparison can be simpler than learning a strong value baseline.
- **Larger groups give stronger comparisons**: but they also require more sampling compute.

### A Limitation of Relative Optimization

A subtle failure mode of relative optimization is that the whole group can collapse to uniformly bad answers. If every sampled response receives the same reward---for example, all are wrong and all get 0---then the normalized within-group signal becomes weak or vanishes. That means GRPO usually works best when the base model is already capable enough to produce some reward diversity, and when group size is large enough to expose meaningful differences.

### A Verifiable Reasoning Example

**One Prompt, Four Candidates**

**Prompt**: `A store sells 3 apples for $6. How much do 5 apples cost at the same rate?`

1. `$10`
2. `$12`
3. `$8`
4. `$9`

If the verifier says `$10` is correct, GRPO compares the four candidates relative to one another and pushes the model toward answers that look more like the winner.

### Minimal Sketch

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

In practice, GRPO is particularly suitable for reasoning datasets with verifiable rewards. A checker can inspect the final `\boxed{}` answer in math tasks, assign reward automatically, and make within-group comparison easier to define and reproduce.

### Limits of GRPO

- **Relative does not mean perfect**: the “best in the group” can still be bad.
- **Reward quality still dominates**: if the verifier is weak or gameable, the optimization target is still flawed.

## Offline and Online Optimization

SFT and DPO are *offline* methods because they learn from fixed datasets. PPO and GRPO are *online* methods because they generate fresh responses during training and update from newly sampled trajectories. Offline optimization is usually simpler and more stable; online optimization is closer to direct policy improvement.

Because online RL updates the model on samples drawn from its current policy, its distribution shift can in some cases be smaller than that of SFT. This is a tendency rather than a guarantee; the actual outcome still depends on reward quality, data coverage, and evaluation.

## Choosing the Right Tool

| Method | Training data                                        | Main signal                                                | Online? | Best fit                                             |
| ------ | ---------------------------------------------------- | ---------------------------------------------------------- | ------- | ---------------------------------------------------- |
| SFT    | prompt + response                                    | imitation                                                  | No      | bootstrapping, format control, instruction following |
| DPO    | prompt + chosen + rejected                           | preference comparison                                      | No      | behavior correction, alignment, preference shaping   |
| PPO    | prompt + sampled responses + reward + value estimate | scalar reward + token-level advantage                      | Yes     | strongest but costly online optimization             |
| GRPO   | prompt + grouped sampled responses                   | group-normalized reward $\rightarrow$ token-wise advantage | Yes     | verifiable reasoning and lower-memory online RL      |

**A Practical Selection Order**

- Move to **DPO** when the issue is not basic capability but preference or response style.**DPO**。

Method selection should begin with the form of supervision actually available. If target answers can be written explicitly, the problem is suited to SFT. If only pairwise ordering is reliable, the problem is suited to DPO. If sampled outputs can be scored by a stable reward pipeline, online RL becomes justified.

## A Worked Example

Consider a small arithmetic example. It is enough to show how the supervision signal becomes richer as we move from SFT to DPO and then to online RL.

### SFT Dataset

```python
sft_data = [
    {"prompt": "2 + 3 =", "response": "5"},
    {"prompt": "7 + 8 =", "response": "15"},
    {"prompt": "9 - 4 =", "response": "5"},
]
```

### DPO Dataset

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

### Online RL Reward

```python
def reward_fn(pred, ground_truth):
    return 1.0 if pred.strip() == ground_truth.strip() else 0.0
```

### GRPO-style Generation

```python
candidates = ["14", "15", "16", "13"]
rewards = [reward_fn(c, "15") for c in candidates]
# -> [0, 1, 0, 0]
```

This example already makes the progression visible. SFT corresponds to matching a target answer; DPO corresponds to preferring one answer over another; online RL corresponds to sampling first and then updating from reward; GRPO corresponds to sampling several answers and updating from their relative comparison.

## Engineering Considerations

- **Data quality beats algorithm cleverness**: a clean SFT or DPO dataset can outperform a fancier pipeline with noisy supervision.
- **Match data to the algorithm**: SFT needs demonstrations, DPO needs pairwise comparisons, and online RL needs prompts plus a reliable reward pipeline.
- **Separate capability from preference**: first ask whether the model *cannot* do the task or whether it merely *chooses the wrong kind of answer*.
- **Prefer verifiable rewards first**: online RL is far safer when reward comes from exact checks instead of vague human impressions alone.
- **Make reward extraction explicit**: if the scorer only looks at a final answer string or a unit-test result, that exact extraction logic should also be part of evaluation and debugging.
- **Track regressions explicitly**: if a model gets better on one task but worse elsewhere, the training pipeline still needs work.
- **Choose the simplest method that matches the signal**: do not jump to online RL if a well-curated SFT or DPO pipeline already solves the problem.

## Conclusion

Across SFT, DPO, PPO, and GRPO, the central issue is not whether the model can predict the next token, but what supervision defines a better continuation. SFT relies on demonstrations, DPO on pairwise preferences, PPO on sampled trajectories with reward and value-function estimates, and GRPO on relative rewards within sampled groups.

Method choice should follow supervision rather than fashion. When target answers are explicit, SFT is usually sufficient. When only relative ordering is reliable, DPO is usually more appropriate. PPO or GRPO become necessary only when reward signals are dependable and online optimization is worth its additional cost. In all cases, data construction, reward design, and evaluation determine whether post-training improves behavior without eroding existing capability.

**Final Takeaway / 最终结论**

*LLM post-training turns next-token prediction into goal-directed sequential decision making; the appropriate method is the one that matches the available supervision with the least unnecessary complexity.*

*LLM 后训练，本质上是把 next-token prediction 转化为面向目标的序列决策；真正合适的方法，是在不过度增加复杂度的前提下，最能匹配现有监督信号的方法。*

## References

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