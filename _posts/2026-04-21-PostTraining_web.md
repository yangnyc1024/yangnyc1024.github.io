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

**From Supervised Fine-Tuning to Reinforcement Learning**

An overview of how post-training reshapes a base model into a usable assistant.

2026-04-21

**Overview**

- **SFT** learns from demonstrations.
- **DPO** learns from preferred-vs-rejected comparisons.
- **PPO** learns online from sampled responses, rewards, and a critic.
- **GRPO** keeps online optimization but replaces the critic with group-relative comparison.
- A useful summary is that post-training turns next-token prediction into a form of goal-directed sequential decision making.

**Reading Guide**

The discussion proceeds from motivation to problem formulation, then to the objectives of SFT, DPO, PPO, and GRPO, and finally to practical method selection.

最简单的记法是四句话。SFT 是模仿，DPO 是比较，PPO 是带 critic 的在线奖励优化，GRPO 是组内相对比较。

## Why Post-Training Matters

Pretraining gives a model fluency, broad knowledge, and the ability to continue text. What it does *not* automatically give is the product behavior we care about: following instructions, staying on format, respecting preferences, behaving safely, or reasoning reliably under constraints.

Post-training is the phase where we reshape a base model into a policy that better matches human goals or task-defined goals. Depending on the available data, that goal may be imitation, pairwise preference alignment, online reward maximization, or success on verifiable tasks such as math, code, or tool use.

**Core intuition**

If pretraining teaches a model to *speak*, post-training teaches it how to *behave*.

## A Unifying Lens: LLMs as Sequential Decision Makers

A clean way to understand post-training is to reinterpret generation as a sequential decision problem. Given a prompt $x$, the model generates a response $y=(y_1, y_2, \dots, y_T)$. At step $t$, the current context $(x, y_{<t})$ is the *state*, the next token $y_t$ is the *action*, and the model distribution $\pi_\theta(y_t \mid x, y_{<t})$ is the *policy*, where $\theta$ denotes the trainable parameters.

Once a full answer is produced, we can assign a reward: helpfulness, correctness, safety, human preference, or a verifiable score from a checker. This is why RL words such as *policy*, *reward*, *return*, *value*, and *advantage* show up so naturally in modern LLM training.

| Concept| LLM interpretation| Why it matters|
| --- | --- | --- |
| State| Prompt + partial response | Decides what information the model can condition on|
| Action| Next token | The smallest unit of decision|
| Policy| $\pi_\theta(y_t \mid x, y_{<t})$ | Controls how likely each continuation is|
| Reward| Preference, correctness, safety, verifier score | Tells the model what “better” means|
| Value| Expected future quality from the current prefix | Helps estimate how promising a partial answer is|
| Trajectory| A full generated response | The object that finally gets evaluated|

## One Prompt, Four Learning Signals

Consider a simple prompt: `What is 12 + 7?` Even this small example is enough to distinguish the four main families of post-training methods.

**Same prompt, different supervision**

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

In this objective:

- $\mathcal{L}_{\mathrm{SFT}}$ is the loss being minimized.
- $N$ is the number of prompt-response pairs.
- $T_i$ is the response length for example $i$.
- The inner sum runs over response tokens.
- The same parameter vector $\theta$ is shared across all examples and token positions.

This is why SFT is often compared to behavior cloning in RL. We do not ask the model to explore. We simply show what good behavior looks like and make it imitate.

### Minimal code

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

### Why SFT works well

- **Bootstrapping**: quickly teaches a new response style, output format, tool schema, or domain convention.
- **Stability**: the objective is simple and well understood.
- **Scalability**: large amounts of curated prompt-response data are easy to pipeline.

### Where SFT falls short

- **No explicit ranking**: it does not tell the model *why* one answer is better than another.
- **No exploration**: the model only learns from fixed demonstrations.
- **Data narrowness**: if the data is too narrow, the model may overfit new patterns and regress elsewhere.

**Best use case**

Use SFT when you already know what the response should look like and can write it down clearly.

## DPO: Learning from Comparisons

Direct Preference Optimization sits between pure imitation and full online RL. Instead of asking the model to copy one target answer, DPO gives a prompt, a chosen response, and a rejected response, then optimizes the model to score the chosen one higher. ([rafailov2023dpo](#ref-rafailov2023dpo))

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

Here:

- $\mathcal{L}_{\mathrm{DPO}}$ is the loss being minimized.
- $\mathcal{D}$ is a preference dataset.
- $y^{+}$ and $y^{-}$ denote the chosen and rejected responses for the same prompt $x$.
- $\pi_{\mathrm{ref}}$ is a frozen reference policy.
- $\beta$ controls how sharply the model separates the two responses.
- $\sigma(\cdot)$ is the sigmoid function.
- $\log \pi_\theta(y\mid x)$ means the log-probability of the whole response sequence $y$, i.e. the sum of token-level log-probabilities across the generated answer.

The key idea is not “imitate this answer,” but “prefer this answer over that one.” Mathematically, DPO is best viewed as a classification-style loss over preference pairs rather than an on-policy RL objective. That makes DPO especially useful when the model is already roughly capable but keeps choosing the wrong tone, preference, stance, refusal pattern, or output style.

**Toy preference pair**

**Prompt**: Explain why regularization helps generalization.

**Chosen**: Regularization discourages overly complex models and reduces overfitting.

**Rejected**: Regularization mainly makes training faster, so it generalizes better.

DPO teaches the model that the first answer should rank above the second.

### Minimal pseudocode

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp_pos, logp_neg, logp_ref_pos, logp_ref_neg, beta=0.1):
    logits = beta * ((logp_pos - logp_ref_pos) - (logp_neg - logp_ref_neg))
    return -F.logsigmoid(logits).mean()
```

This is not production code, but it captures the core structure: compare the preferred and rejected answers relative to a frozen reference model, then push the margin in favor of the chosen answer.

### A practical rewrite example

A common DPO use case is identity or style rewriting. Suppose a model originally answers “I'm Qwen.” For the same identity prompt, we mark that original answer as rejected and a lightly edited answer such as “I'm Deep Qwen.” as chosen. DPO does not need to relearn all model behavior; it only needs to shift preference between competing responses.

**Interpretation**

This example is best understood as local behavioral correction. In practice, tools such as `DPOTrainer`, `DPOConfig`, and the hyperparameter $\beta$ mainly control how strongly the model separates preferred and rejected responses.

### Why DPO is appealing

- **Preference-aware**: it directly encodes relative quality.
- **No separate reward model**: the optimization is written directly in terms of log-probability differences.
- **Operationally simpler than online RL**: no on-policy sampling loop is required during training.

### Its limitations

- **Still offline**: the model does not learn from its current sampled behavior.
- **Depends on comparison quality**: noisy chosen-vs-rejected pairs can teach the wrong boundary.

## PPO: Online RL with a Critic

The big conceptual jump from SFT and DPO to PPO is that training becomes *online*. The model is no longer updated only from a frozen dataset; it generates fresh responses under its current policy, those responses are scored, and the model is updated using that newly collected experience. ([schulman2017ppo](#ref-schulman2017ppo); [ouyang2022instructgpt](#ref-ouyang2022instructgpt))

**Typical PPO loop for LLMs**

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

Here $\mathcal{J}^{\mathrm{CLIP}}_{\mathrm{PPO}}$ is written as a surrogate objective to maximize; in implementation, one usually minimizes its negative. For the notation:

- $\mathbb{E}_t[\cdot]$ is the expectation over sampled time steps.
- $s_t$ and $a_t$ denote the state and action at step $t$.
- $\theta_{\mathrm{old}}$ is the policy used to collect the current batch of trajectories.
- $\hat{A}_t$ is an estimated advantage.
- $\epsilon$ is the clipping radius.
- $\operatorname{clip}(\cdot)$ restricts the ratio to a small interval around 1.

Crucially, $r_t(\theta)$ here is the new-to-old policy probability ratio, not the reward itself; reward information enters the update through $\hat{A}_t$.

Conceptually, PPO is best read as a clipped *policy-improvement term*. In full PPO implementations, one usually minimizes the negative objective above while also including a value loss and sometimes an entropy bonus; in LLM fine-tuning, KL regularization to a reference policy is also common.

The clipping term is what makes PPO practical. It allows policy improvement while preventing each update from moving too far from the previous policy. In LLM systems, a reference model or KL penalty is also commonly used to stop the model from drifting too aggressively.

### A short intuition example

Consider the prompt “Write a concise answer: what is the capital of France?” One sample says “Paris” and another says “Berlin.” PPO does not only learn that one final answer is right and the other is wrong. It asks a finer question: which token decisions along the way pushed the response toward a better outcome? That is exactly where token-level advantage becomes useful.

### Why value matters

Rewards for language tasks are often sparse and noisy. A value model estimates the expected future return from a partial prefix, which helps reduce variance and gives finer credit assignment across the generated tokens.

### Minimal pseudocode

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

### Two common reward styles

- **Learned reward**: a reward model trained from human preference data.
- **Verifiable reward**: exact answer matching, unit tests, tool success, structured constraints, or other deterministic checkers.

### Trade-offs

- **Strong but expensive**: PPO can be powerful, but it needs more infrastructure and more memory.
- **Sensitive to reward quality**: a weak reward signal can train the wrong behavior very efficiently.

## GRPO: Critic-Free Relative Optimization

GRPO remains an online RL method, but removes the separate value model. Instead of using a critic to estimate future return, it samples a group of responses for the same prompt and computes *relative* advantages inside the group. ([shao2024deepseekmath](#ref-shao2024deepseekmath))

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

Here $\mathcal{J}^{\mathrm{simple}}_{\mathrm{GRPO}}$ is again written as an objective to maximize. To keep the notation readable:

- $G$ is the group size.
- $y_i$ is the $i$-th sampled response.
- $|y_i|$ is the length of response $y_i$.
- $y_{i,t}$ is the $t$-th token in response $y_i$.
- $r_i$ is the scalar reward assigned to the whole response $y_i$.
- $\bar r$ is the mean reward within the group.
- $\operatorname{std}(r_1,\dots,r_G)$ is the within-group standard deviation.
- $\epsilon$ is the PPO-style clipping radius.
- $\varepsilon$ is a small constant for numerical stability.

In this simplified presentation, every token in response $y_i$ shares the same response-level signal $\tilde r_i$. In other words, GRPO uses $\rho_{i,t}$ for the policy ratio and reserves $r_i$ for reward.

A useful intuition is the following: if one sampled answer is better than the others in the same group, increase its probability; if it is worse, decrease it. The optimization is still token-wise, but the advantage now comes from group-relative normalized rewards rather than a learned critic.

> 中文提示：GRPO 最适合用一句话记住。它还是做 token-level policy update，但 advantage 不再靠 value model，而是靠组内相对奖励。

**Practical Features of GRPO**

- **No value model**: lower memory footprint and less system complexity.
- **Natural for verifiable tasks**: math, code, structured reasoning, and tool-driven tasks benefit from clear relative scoring.
- **Good when critic training is unstable**: relative comparison can be simpler than learning a strong value baseline.

### A math example

**One prompt, four candidates**

**Prompt**: `A store sells 3 apples for 6 dollars. How much do 5 apples cost at the same rate?`

1. `10 dollars`
2. `12 dollars`
3. `8 dollars`
4. `9 dollars`

If the verifier says `10 dollars` is correct, GRPO compares the four candidates relative to one another and pushes the model toward answers that look more like the winner.

### Minimal sketch

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

### What to watch out for

- **Relative does not mean perfect**: the “best in the group” can still be bad.
- **Reward quality still dominates**: if the verifier is weak or gameable, the optimization target is still flawed.

## Offline vs Online: Why the Distinction Matters

SFT and DPO are *offline* methods because they learn from a fixed dataset. PPO and GRPO are *online* methods because they generate fresh responses during training and update from those newly generated samples. This distinction matters because offline methods are usually simpler and more stable, while online methods are closer to true policy optimization.

One useful intuition---not a universal law---is that online RL can sometimes be less distribution-shifting because it updates the model on samples drawn from its current policy. By contrast, SFT may pull the model toward target outputs that are farther from its current distribution. Whether this preserves capability better depends heavily on reward design, data coverage, and evaluation.

**Summary**

Offline methods optimize against a fixed snapshot of behavior; online methods optimize against the model's current behavior.

## Choosing the Right Tool

| Method| Training data| Main signal| Online?| Best fit|
| --- | --- | --- | --- | --- |
| SFT | prompt + response | imitation| No| bootstrapping, format control, instruction following|
| DPO | prompt + chosen + rejected | preference comparison| No| behavior correction, alignment, preference shaping|
| PPO | prompt + sampled responses + reward + value estimate | scalar reward + token-level advantage| Yes| strongest but costly online optimization|
| GRPO | prompt + grouped sampled responses | group-normalized reward $\rightarrow$ token-wise advantage| Yes| verifiable reasoning and lower-memory online RL|

**A Common Selection Order**

- Start with **SFT** if you have good demonstrations.
- Move to **DPO** when the issue is not basic capability but preference or response style.
- Use **PPO** or **GRPO** only when reward is trustworthy and the infrastructure is ready.


## Why RL Vocabulary Is Useful

The RL terms used in LLM post-training are not merely stylistic borrowings. *Return* is the total reward of a sampled response. *Value* estimates expected future return from a prefix. *Advantage* measures how much better a sampled action was than a baseline expectation. ([sutton2018rl](#ref-sutton2018rl))

This is also where the connection to dynamic programming becomes clearer. PPO carries the flavor of approximate dynamic programming because it learns a value function to support policy improvement. GRPO removes that value function, so it feels more Monte-Carlo-like and less like explicit value-based estimation. This is best read as intuition, not as an exact equivalence between modern LLM training and classical tabular RL. ([sutton2018rl](#ref-sutton2018rl); [schulman2017ppo](#ref-schulman2017ppo); [shao2024deepseekmath](#ref-shao2024deepseekmath))

The clean classical RL template behind this discussion is the MDP tuple $(S, A, P, R, \gamma)$. LLM post-training does not literally solve a tabular MDP, but the mapping remains useful as a mental model: prompt plus prefix is the state, the next token is the action, appending a token is the transition, and response quality is the reward. ([sutton2018rl](#ref-sutton2018rl))

$$
(S, A, P, R, \gamma).
$$

In the classical MDP tuple, $S$ is the state space, $A$ is the action space, $P$ is the transition kernel, $R$ is the reward function, and $\gamma$ is the discount factor. These symbols are introduced here only as a conceptual bridge; the later LLM formulas are not direct tabular MDP updates.

**MDP components mapped to LLM training**

| State| prompt + partial response|
| --- | --- |
| Action| next token|
| Transition| append one token and move to a new prefix|
| Reward| helpfulness, correctness, safety, verifier score|
| Discount| often omitted or absorbed into finite-horizon return definitions|

**Intuition Only**

**Q-learning intuition**: “If I choose this token, what is the best continuation I could get?”

**SARSA intuition**: “If I choose this token, what continuation will I get under my current behavior?”

These analogies are useful for thinking, even though modern LLM post-training does not literally run tabular Q-learning or SARSA.

### Q-learning and SARSA formulas

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[
r + \gamma \max_{a'} Q(s',a') - Q(s,a)
\Big].
$$

Q-learning imagines the *best continuation* available after the current action. In LLM language, the analogy is: if I choose this token now, what is the best continuation I could still reach from here? ([watkins1992qlearning](#ref-watkins1992qlearning); [sutton2018rl](#ref-sutton2018rl))

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[
r + \gamma Q(s',a') - Q(s,a)
\Big].
$$

In both update rules, $\alpha$ is the learning rate, $\gamma$ is the discount factor, and $s'$ denotes the next state. The difference lies in the treatment of the next action: Q-learning uses $\max_{a'}Q(s',a')$, while SARSA uses the value of the action $a'$ actually chosen by the current behavior policy. ([watkins1992qlearning](#ref-watkins1992qlearning); [rummery1994online](#ref-rummery1994online); [sutton2018rl](#ref-sutton2018rl))

SARSA instead follows the current behavior policy. In LLM language, that sounds like: if I choose this token now, what continuation will my current model actually keep generating, rather than the best imaginable one? ([rummery1994online](#ref-rummery1994online); [sutton2018rl](#ref-sutton2018rl))

**Connecting Classical RL to Post-Training**

- **Q-learning**: think in terms of the best continuation.
- **SARSA**: think in terms of the continuation under the current behavior.
- **PPO**: think in terms of advantage-weighted continuation with a learned baseline.
- **GRPO**: think in terms of relative continuation inside a sampled group.

## A Small End-to-End Example

Consider a small arithmetic example. It is enough to show how the supervision signal becomes richer as we move from SFT to DPO and then to online RL.

### SFT dataset

```python
sft_data = [
    {"prompt": "2 + 3 =", "response": "5"},
    {"prompt": "7 + 8 =", "response": "15"},
    {"prompt": "9 - 4 =", "response": "5"},
]
```

### DPO dataset

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

### Online RL reward

```python
def reward_fn(pred, ground_truth):
    return 1.0 if pred.strip() == ground_truth.strip() else 0.0
```

### GRPO-style generation

```python
candidates = ["14", "15", "16", "13"]
rewards = [reward_fn(c, "15") for c in candidates]
# -> [0, 1, 0, 0]
```

This example already makes the progression visible. SFT corresponds to matching a target answer; DPO corresponds to preferring one answer over another; online RL corresponds to sampling first and then updating from reward; GRPO corresponds to sampling several answers and updating from their relative comparison.

## Engineering Considerations

- **Data quality beats algorithm cleverness**: a clean SFT or DPO dataset can outperform a fancier pipeline with noisy supervision.
- **Separate capability from preference**: first ask whether the model *cannot* do the task or whether it merely *chooses the wrong kind of answer*.
- **Prefer verifiable rewards first**: online RL is far safer when reward comes from exact checks instead of vague human impressions alone.
- **Track regressions explicitly**: if a model gets better on one task but worse elsewhere, the training pipeline still needs work.
- **Choose the simplest method that matches the signal**: do not jump to online RL if a well-curated SFT or DPO pipeline already solves the problem.

**Practical Guideline**

A common workflow is: **use SFT to establish basic behavior**, **use DPO to adjust preferences**, and introduce **PPO/GRPO only when reward is reliable and online optimization is necessary**.

## Condensed Summary

A compact comparison is as follows: SFT improves behavior by imitation, DPO by pairwise comparison, PPO by online sampling with reward and a critic, and GRPO by online sampling with group-relative comparison instead of a critic. From the MDP viewpoint, all four methods improve the policy of a token-generating agent, but they differ in the learning signal and optimization structure they use.

**A Slightly More Formal Comparison**

SFT is closest to behavior cloning, DPO is offline preference optimization with an implicit reward view, PPO looks like approximate policy improvement with value estimation, and GRPO is a critic-free relative policy optimization method that is especially attractive for verifiable reasoning tasks.

## Conclusion

A short concluding summary is this: SFT improves behavior through demonstration matching, DPO through preference comparison, PPO through online reward optimization with a critic, and GRPO through online optimization with group-relative comparison.

At a coarse level, SFT provides demonstrations, DPO provides pairwise judgments, PPO provides scalar rewards with a learned baseline, and GRPO provides group-relative reward comparisons.

A compact way to connect the whole discussion is: *LLM post-training turns next-token prediction into a form of goal-directed sequential decision making.*

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
