---
title: "Attention in LLMs: From Intuition to Kernel View, with PyTorch"
subtitle: "A practical walkthrough from intuition to equations to implementation."
date: 2026-04-20 12:00:00 +0800
categories:
  - LLM
  - Attention
  - PyTorch
tags:
  - attention
  - transformer
  - kernel regression
  - self-attention
description: "A practical explanation of Transformer attention, from intuition and QKV roles to kernel regression and PyTorch code."
---

## 0. Introduction

Attention is the core mechanism behind modern large language models. It allows each token to dynamically decide which other tokens matter, instead of forcing information to flow only through a fixed sequential chain as in RNNs. This is one of the main reasons Transformers can model long-range dependencies much more effectively and train in parallel.  

A lot of explanations stop at saying that attention is “just a weighted average.” That statement is not wrong, but it is incomplete. A deeper view is that attention separates **matching** from **content aggregation**: queries and keys define how relevance is computed, while values define what information is actually passed forward. At an even deeper level, attention can be interpreted as a learned kernel regression operating in latent spaces.  

In this blog, we will move step by step from intuition to mathematics to PyTorch implementation. Instead of putting code at the very end as an appendix, we will integrate code directly into the explanation, so that each formula corresponds to an actual implementation detail.  

---

# 1. Why Attention Exists

Before attention, sequence modeling was dominated by recurrent neural networks such as RNNs, LSTMs, and GRUs. These models process tokens one by one, updating a hidden state over time. This sequential structure gives them a natural notion of order, but it also introduces an information bottleneck: all previous information must be compressed into the current hidden state.  

As the sequence becomes longer, it becomes increasingly hard for the model to preserve important details from far-away positions. Even with gating mechanisms in LSTMs, the information still has to travel through many steps. In practice, this makes long-range dependency modeling difficult.  

Attention changes the problem formulation. Instead of asking the model to compress the entire past into one hidden vector, we allow the current token to directly retrieve information from all tokens, with learned relevance weights.  

That change from **compression** to **retrieval** is the key conceptual shift.  

---

# 2. The Core Attention Formula

The standard scaled dot-product attention is defined as:  

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

At first glance, this formula looks compact, but it hides several different ideas inside one line. The best way to understand it is to decompose it into stages.  

---

## 2.1 Step 1: Compute Similarity Scores

The product $QK^\top$ computes pairwise similarities between queries and keys. If token $i$ has query $q_i$ and token $j$ has key $k_j$, then their raw score is:  

$$
s_{ij} = q_i^\top k_j
$$

This dot product is not “the information itself.” It is a compatibility score. It tells us how strongly token $i$ should attend to token $j$.  

---

## 2.2 Step 2: Normalize the Scores

The raw scores are then scaled and passed through softmax:  

$$
\alpha_{ij} = \frac{\exp(s_{ij} / \sqrt{d_k})}{\sum_{m} \exp(s_{im} / \sqrt{d_k})}
$$

This turns arbitrary similarity scores into attention weights that sum to 1 across all keys for each query.  

Softmax is important because it creates competition. If one key becomes much more relevant, its weight increases while others decrease. The model is forced to prioritize.  

---

## 2.3 Step 3: Aggregate Values

Finally, attention uses the weights to compute a weighted sum of values:  

$$
o_i = \sum_j \alpha_{ij} v_j
$$

So the output for token $i$ is not copied from one position. It is an adaptive combination of information from many positions.  

---

# 3. Q, K, V: What They Really Mean

A lot of people memorize “Q is query, K is key, V is value,” but that alone does not yet explain why attention works so well.  

The deeper point is that Q, K, and V are three different learned projections of the same input representation. For an input token vector $x$, we compute:  

$$
q = W_Q x,\qquad k = W_K x,\qquad v = W_V x
$$

These are not arbitrary copies. They serve different roles.  

---

## 3.1 Query: What Am I Looking For?

The query represents the “search intent” of the current token. It asks what kind of information this token needs from the rest of the sequence.  

For example, if the current token is a verb, it may need information about its subject. If the current token is a pronoun, it may need information about its antecedent. The query encodes that demand.  

So query is not a content vector in the usual sense. It is more like a request vector.  

---

## 3.2 Key: How Can I Be Matched?

The key represents how each token advertises itself to other tokens. It determines how this token can be found by a query.  

A key is therefore not primarily about passing information forward. Its role is to participate in the similarity function.  

A useful intuition is:  

- Query says: “What do I need?”  
- Key says: “What kind of token am I?”  

---

## 3.3 Value: What Information Do I Provide?

The value is the actual information that will be aggregated once a token is selected by attention.  

This is extremely important. Attention weights are computed from Q and K, but the output comes from V. That means matching and content are explicitly separated.  

This separation makes the mechanism more expressive. A token can be highly matchable in one learned space while contributing information in another learned space.  

---

## 3.4 Why Not Use Just One Projection?

A natural question is: why not directly compute attention from one representation?  

Because the tasks of **finding relevant tokens** and **representing their content** are not the same.  

If the same vector had to simultaneously serve as query, key, and value, then the model would be forced to use one space for three different purposes: asking, matching, and carrying information. That is unnecessarily restrictive.  

By learning separate projections, the model gains flexibility: it can learn one space optimized for similarity and another space optimized for content representation.  

这一节最关键。Q 决定“我要找什么”，K 决定“我像什么”，V 决定“我真正提供什么内容”。

---

# 4. PyTorch Implementation of Single-Head Self-Attention

Now let us move from equations to code. A good way to build intuition is to implement single-head self-attention from scratch.  

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Optional tensor broadcastable to (batch_size, seq_len, seq_len)
                  where masked positions are 0 and visible positions are 1

        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        # 1) Linear projections
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)  # (B, T, D)
        V = self.W_v(x)  # (B, T, D)

        # 2) Similarity scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)  # (B, T, T)

        # 3) Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 4) Normalize into attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # 5) Weighted aggregation of values
        output = torch.matmul(attn_weights, V)  # (B, T, D)

        return output
```

This implementation is short, but almost every line maps directly to the theory.

------

## 4.1 The Input Shape

The input `x` has shape `(batch_size, seq_len, d_model)`.

That means for each batch, we have a sequence of token representations, and each token is represented by a `d_model`-dimensional vector.

------

## 4.2 Linear Projections in Code

These lines implement the projection equations:

```
Q = self.W_q(x)
K = self.W_k(x)
V = self.W_v(x)
```

Mathematically, they correspond to:
$$
Q = X W_Q,\qquad K = X W_K,\qquad V = X W_V
$$
In practice, PyTorch’s `nn.Linear` applies the transformation to the last dimension. So for each token representation, it produces a new projected vector.

------

## 4.3 Score Computation in Code

The line

```
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
```

computes all pairwise dot products between queries and keys.

If `Q` has shape `(B, T, D)` and `K` has shape `(B, T, D)`, then `K.transpose(-2, -1)` has shape `(B, D, T)`, so the result has shape `(B, T, T)`.

That `(T, T)` matrix for each batch tells us: for each token position, how strongly it attends to every token position in the sequence.

------

## 4.4 Softmax in Code

The line

```
attn_weights = F.softmax(scores, dim=-1)
```

normalizes along the last dimension, which means that for each query token, the weights across all key tokens sum to 1.

So each row of the attention matrix is a probability distribution over source positions.

------

## 4.5 Weighted Sum of Values in Code

The last line

```
output = torch.matmul(attn_weights, V)
```

takes the weighted sum of values. Since `attn_weights` has shape `(B, T, T)` and `V` has shape `(B, T, D)`, the output has shape `(B, T, D)`.

This means every output token is a contextualized representation computed from the whole sequence.

------

# 5. A Concrete Mini Example

Let us run a small example so the shapes become concrete.

```python
torch.manual_seed(42)

B, T, D = 2, 4, 8
x = torch.randn(B, T, D)

attn = SelfAttention(d_model=D)
y = attn(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
```

Expected output:

```python
Input shape : torch.Size([2, 4, 8])
Output shape: torch.Size([2, 4, 8])
```

The shape stays the same because attention transforms each token representation into a new contextualized token representation of the same dimension.

------

# 6. Why Masking Matters

In practice, attention often needs masking.

There are two major reasons.

First, some positions may be padding tokens and should not contribute information.

Second, in autoregressive language models like GPT, a token should not attend to future tokens.

------

## 6.1 Causal Mask in PyTorch

Here is how to build a standard causal mask:

```python
def causal_mask(seq_len: int) -> torch.Tensor:
    """
    Returns a lower-triangular mask of shape (1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)
```

Usage:

```python
mask = causal_mask(T)
y = attn(x, mask=mask)
```

This lower-triangular mask ensures that position $i$ can only attend to positions up to $i$, not after it.

------

# 7. Multi-Head Attention

Single-head attention already works, but Transformer uses multi-head attention because one similarity space is often not enough. Different relationships may require different projection subspaces.

For example, one head may focus on syntactic structure, another on long-range reference, and another on local phrase composition.

Instead of forcing all these patterns into one Q/K/V space, we learn several smaller Q/K/V spaces in parallel.

------

## 7.1 PyTorch Implementation of Multi-Head Attention

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: Optional tensor broadcastable to (B, H, T, T) or (1, 1, T, T)

        Returns:
            output: (B, T, D)
        """
        B, T, D = x.shape

        # 1) Linear projections
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)  # (B, T, D)
        V = self.W_v(x)  # (B, T, D)

        # 2) Split into heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Hd)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Hd)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Hd)

        # 3) Compute scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
        context = torch.matmul(attn_weights, V)   # (B, H, T, Hd)

        # 4) Merge heads
        context = context.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        # 5) Final output projection
        output = self.W_o(context)  # (B, T, D)

        return output
```

This is the full multi-head version, and it is still conceptually the same pipeline as single-head attention.

------

## 7.2 What Changes from Single-Head to Multi-Head?

The main difference is that after projection, we split the representation dimension into multiple heads.

If `d_model = 512` and `num_heads = 8`, then each head works in a 64-dimensional space.

Each head computes its own attention pattern independently.

After that, all head outputs are concatenated and then passed through a final linear layer `W_o`.

------

## 7.3 Why the Final Output Projection `W_o`?

A common question is: if we already concatenated all heads, why do we still need another linear layer?

Because concatenation only stacks the information. It does not mix or re-coordinate the different subspaces.

The final projection learns how to combine information across heads and map it back into the model space.

------

# 8. Connecting the Code Back to Theory

Now that we have seen the code, it is useful to reconnect it to the conceptual picture.

In the code, the projections `W_q`, `W_k`, and `W_v` are the learned feature maps. They transform the raw token representation into three different latent spaces.

The operation `Q @ K^T` computes a similarity structure in the learned query-key space.

The multiplication by `V` performs information aggregation in the learned value space.

So the code reflects a very clean division of labor:

- Q/K define **where information should flow**
- V defines **what information should flow**

------

# 9. Attention as Kernel Regression

This brings us to the deeper statistical interpretation.

Classical kernel regression has the form:
$$
f(x_i) = \sum_j K(x_i, x_j) f(x_j)
$$
This says that the estimate at point $x_i$ is a weighted combination of function values at other points, where the weights are determined by a kernel measuring similarity.

Attention has almost the same structure:
$$
\mathrm{Attn}(x_i) = \sum_j \alpha_{ij} v_j
$$
where the attention weights $\alpha_{ij}$ come from query-key similarity.

So we can interpret:

- $QK^\top$ defines a learned kernel
- $V$ provides the learned function values

如果你想把这篇文章和统计学习联系起来，就重点看这里。attention 可以看成在隐空间里做一次可学习的核回归。

This is why it is reasonable to say:

> Attention is kernel smoothing in a learned latent space.

------

# 10. Why This Interpretation Matters

This kernel view is not just philosophical. It helps explain why attention is so flexible.

In classical machine learning, the choice of kernel determines how similarity is measured. In attention, the model does not use a fixed kernel like RBF. Instead, it learns the feature maps that induce the kernel through $W_Q$ and $W_K$.

At the same time, $W_V$ learns what information should be passed around once similarity is established.

So Transformer is jointly learning two things:

1. a similarity structure
2. a content representation

That is a very powerful combination.

------

# 11. Interview-Level Summary

If you need a concise but high-quality explanation in an interview, here is a strong version:

Attention computes a weighted aggregation of value vectors, where the weights are determined by learned similarity between queries and keys. Queries encode what information a token is looking for, keys encode how tokens can be matched, and values encode the information being retrieved. From a statistical perspective, attention can be interpreted as a learned kernel regression in latent space.

------

# 12. Final Takeaways

Attention is not merely a weighted average. It is a structured mechanism that separates relevance estimation from information transport. That separation is what makes it expressive, scalable, and suitable for modern large language models.

The PyTorch code makes this especially clear: the model first learns Q/K/V projections, then computes similarities, then normalizes them, and finally aggregates values. Every line of code reflects a conceptual decision.

The deeper kernel view gives an even stronger unification: attention learns both the kernel and the function representation at the same time.

That is why attention is not just useful engineering. It is also a beautiful statistical idea.

------

# 13. Full PyTorch Demo

For completeness, here is a runnable example combining multi-head attention and a causal mask.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_mask(seq_len: int) -> torch.Tensor:
    """
    Returns shape (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(B, T, D)
        output = self.W_o(context)

        return output

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, D = 2, 5, 16
    H = 4

    x = torch.randn(B, T, D)
    mask = causal_mask(T)

    layer = MultiHeadSelfAttention(d_model=D, num_heads=H)
    y = layer(x, mask=mask)

    print("Input shape :", x.shape)
    print("Mask shape  :", mask.shape)
    print("Output shape:", y.shape)
```

This example is already close to the core of what happens inside a Transformer block, aside from residual connections, layer normalization, and the feed-forward network.

------

# 14. One-Sentence Summary

Attention is a learned retrieval mechanism in which Q and K define relevance, V defines content, and the whole computation can be viewed both as neural message passing and as kernel regression in latent space.

# References

\- Hands-On Large Language Models  
\- Build a Large Language Model from Scratch  
\- StatQuest (YouTube)  
\- Datawhale Happy LLM  
\- https://zhuanlan.zhihu.com/p/626820422  
\- https://www.zhihu.com/question/298810062  
