---
title: "Attention in LLMs: From Intuition to Kernel View, with PyTorch"
subtitle: "大语言模型中的 Attention：从直觉到核方法视角，并结合 PyTorch 代码讲解"
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
description: "从直觉到核方法视角，用 PyTorch 代码解读 Transformer Attention 的核心机制。"
---

## 0. Introduction

Attention is the core mechanism behind modern large language models. It allows each token to dynamically decide which other tokens matter, instead of forcing information to flow only through a fixed sequential chain as in RNNs. This is one of the main reasons Transformers can model long-range dependencies much more effectively and train in parallel.  
Attention 是现代大语言模型的核心机制。它让每个 token 都能动态决定“哪些其他 token 对自己重要”，而不是像 RNN 那样只能让信息沿着固定的时序链传播。这也是 Transformer 能更好建模长距离依赖、并且支持并行训练的关键原因之一。

A lot of explanations stop at saying that attention is “just a weighted average.” That statement is not wrong, but it is incomplete. A deeper view is that attention separates **matching** from **content aggregation**: queries and keys define how relevance is computed, while values define what information is actually passed forward. At an even deeper level, attention can be interpreted as a learned kernel regression operating in latent spaces.  
很多解释会停留在“attention 就是加权平均”这一层。这并不算错，但并不完整。更深入地看，attention 把**匹配**和**信息聚合**分开了：query 和 key 负责定义“相关性怎么计算”，而 value 负责定义“真正被传递的内容是什么”。再进一步，attention 还可以被理解为一个在隐空间中进行的、可学习的核回归过程。

In this blog, we will move step by step from intuition to mathematics to PyTorch implementation. Instead of putting code at the very end as an appendix, we will integrate code directly into the explanation, so that each formula corresponds to an actual implementation detail.  
这篇文章会从直觉、数学，再到 PyTorch 实现，逐步搭建完整理解。这里不会把代码放到最后当附录，而是把代码直接融合进讲解中，让每一个公式都能对应到真实的实现细节。

---

# 1. Why Attention Exists

Before attention, sequence modeling was dominated by recurrent neural networks such as RNNs, LSTMs, and GRUs. These models process tokens one by one, updating a hidden state over time. This sequential structure gives them a natural notion of order, but it also introduces an information bottleneck: all previous information must be compressed into the current hidden state.  
在 attention 出现之前，序列建模主要由 RNN、LSTM、GRU 这样的循环神经网络主导。这类模型按时间顺序逐个处理 token，并不断更新 hidden state。这样的结构天然带有顺序信息，但同时也带来了信息瓶颈：所有历史信息都必须被压缩进当前 hidden state 里。

As the sequence becomes longer, it becomes increasingly hard for the model to preserve important details from far-away positions. Even with gating mechanisms in LSTMs, the information still has to travel through many steps. In practice, this makes long-range dependency modeling difficult.  
随着序列变长，模型越来越难保留来自很远位置的重要信息。即使 LSTM 引入了门控机制，信息仍然需要经过很多步才能传到当前位置。在实际中，这使得长距离依赖建模非常困难。

Attention changes the problem formulation. Instead of asking the model to compress the entire past into one hidden vector, we allow the current token to directly retrieve information from all tokens, with learned relevance weights.  
Attention 改变了问题的建模方式。它不再要求模型把整个历史压缩进一个 hidden vector，而是允许当前 token 直接从所有 token 中检索信息，并通过学习到的相关性权重来决定取多少。

That change from **compression** to **retrieval** is the key conceptual shift.  
这种从**压缩**到**检索**的转变，是 attention 最关键的思想变化。

---

# 2. The Core Attention Formula

The standard scaled dot-product attention is defined as:  
标准的 scaled dot-product attention 定义为：

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

At first glance, this formula looks compact, but it hides several different ideas inside one line. The best way to understand it is to decompose it into stages.  
这个公式看起来很紧凑，但实际上一行里压缩了好几个不同的思想。理解它的最好方式，是把它拆成几个阶段。

---

## 2.1 Step 1: Compute Similarity Scores

The product \(QK^\top\) computes pairwise similarities between queries and keys. If token \(i\) has query \(q_i\) and token \(j\) has key \(k_j\), then their raw score is:  
\(QK^\top\) 这一步在计算 query 和 key 的两两相似度。如果 token \(i\) 的 query 是 \(q_i\)，token \(j\) 的 key 是 \(k_j\)，那么它们之间的原始分数就是：

\[
s_{ij} = q_i^\top k_j
\]

This dot product is not “the information itself.” It is a compatibility score. It tells us how strongly token \(i\) should attend to token \(j\).  
这个点积并不是“信息本身”，而是一个匹配分数。它告诉我们：token \(i\) 应该多大程度上关注 token \(j\)。

---

## 2.2 Step 2: Normalize the Scores

The raw scores are then scaled and passed through softmax:  
然后这些原始分数会被缩放并送入 softmax：

\[
\alpha_{ij} = \frac{\exp(s_{ij} / \sqrt{d_k})}{\sum_{m} \exp(s_{im} / \sqrt{d_k})}
\]

This turns arbitrary similarity scores into attention weights that sum to 1 across all keys for each query.  
这一步把任意的相似度分数变成注意力权重，并保证对于每个 query 来说，所有 key 的权重之和为 1。

Softmax is important because it creates competition. If one key becomes much more relevant, its weight increases while others decrease. The model is forced to prioritize.  
Softmax 很重要，因为它制造了竞争关系。如果某个 key 更相关，它的权重就会上升，而其他 key 的权重会下降。模型因此被迫做优先级排序。

---

## 2.3 Step 3: Aggregate Values

Finally, attention uses the weights to compute a weighted sum of values:  
最后，attention 会用这些权重对 value 做加权求和：

\[
o_i = \sum_j \alpha_{ij} v_j
\]

So the output for token \(i\) is not copied from one position. It is an adaptive combination of information from many positions.  
所以，token \(i\) 的输出并不是简单从某个位置拷贝来的，而是从多个位置的信息中自适应组合而成的。

---

# 3. Q, K, V: What They Really Mean

A lot of people memorize “Q is query, K is key, V is value,” but that alone does not yet explain why attention works so well.  
很多人会记住“Q 是 query，K 是 key，V 是 value”，但这还不足以解释 attention 为什么这么有效。

The deeper point is that Q, K, and V are three different learned projections of the same input representation. For an input token vector \(x\), we compute:  
更深层的关键在于：Q、K、V 是同一个输入表示经过三组不同的线性投影后得到的。对于一个输入 token 向量 \(x\)，我们计算：

\[
q = W_Q x,\qquad k = W_K x,\qquad v = W_V x
\]

These are not arbitrary copies. They serve different roles.  
它们不是随便复制出来的三个版本，而是承担不同角色。

---

## 3.1 Query: What Am I Looking For?

The query represents the “search intent” of the current token. It asks what kind of information this token needs from the rest of the sequence.  
Query 表示当前 token 的“搜索意图”。它在问：这个 token 需要从其他位置获取什么样的信息？

For example, if the current token is a verb, it may need information about its subject. If the current token is a pronoun, it may need information about its antecedent. The query encodes that demand.  
例如，如果当前 token 是一个动词，它可能需要主语的信息；如果当前 token 是一个代词，它可能需要先行词的信息。query 编码的就是这种“需求”。

So query is not a content vector in the usual sense. It is more like a request vector.  
所以，query 并不是通常意义上的内容表示，它更像是一个“请求向量”。

---

## 3.2 Key: How Can I Be Matched?

The key represents how each token advertises itself to other tokens. It determines how this token can be found by a query.  
Key 表示每个 token 如何向其他 token “展示自己”。它决定这个 token 会以什么方式被 query 匹配到。

A key is therefore not primarily about passing information forward. Its role is to participate in the similarity function.  
因此，key 的主要职责不是把信息往后传，而是参与相似度计算。

A useful intuition is:  
一个很有帮助的直觉是：

- Query says: “What do I need?”  
- Key says: “What kind of token am I?”  

- Query 在说：“我需要什么？”  
- Key 在说：“我属于哪类信息？”  

---

## 3.3 Value: What Information Do I Provide?

The value is the actual information that will be aggregated once a token is selected by attention.  
Value 才是真正会在 attention 中被聚合的信息。

This is extremely important. Attention weights are computed from Q and K, but the output comes from V. That means matching and content are explicitly separated.  
这点非常重要。Attention 的权重是通过 Q 和 K 算出来的，但最终输出的是 V。这意味着“匹配”和“内容”被显式分离了。

This separation makes the mechanism more expressive. A token can be highly matchable in one learned space while contributing information in another learned space.  
这种分离让机制更有表达能力。一个 token 可以在某个学习到的空间里很容易被匹配到，但在另一个学习到的空间里提供的是不同的信息内容。

---

## 3.4 Why Not Use Just One Projection?

A natural question is: why not directly compute attention from one representation?  
一个自然的问题是：为什么不直接只用一个表示来做 attention？

Because the tasks of **finding relevant tokens** and **representing their content** are not the same.  
因为**找到相关 token**和**表示这些 token 的内容**本来就不是同一个任务。

If the same vector had to simultaneously serve as query, key, and value, then the model would be forced to use one space for three different purposes: asking, matching, and carrying information. That is unnecessarily restrictive.  
如果同一个向量同时承担 query、key 和 value 的功能，那么模型就被迫在同一个空间里同时完成“提问”“匹配”“携带信息”三件事，这会非常受限。

By learning separate projections, the model gains flexibility: it can learn one space optimized for similarity and another space optimized for content representation.  
通过学习分开的投影，模型获得了更大的灵活性：它可以学习一个更适合做相似度计算的空间，以及另一个更适合做信息表示的空间。

---

# 4. PyTorch Implementation of Single-Head Self-Attention

Now let us move from equations to code. A good way to build intuition is to implement single-head self-attention from scratch.  
接下来我们从公式走到代码。建立直觉的一个好方法，是先手写一个单头 self-attention。

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
 这个实现虽然很短，但几乎每一行都能直接对应到前面的理论。

------

## 4.1 The Input Shape

The input `x` has shape `(batch_size, seq_len, d_model)`.
 输入 `x` 的形状是 `(batch_size, seq_len, d_model)`。

That means for each batch, we have a sequence of token representations, and each token is represented by a `d_model`-dimensional vector.
 这表示在每个 batch 中，我们有一串 token 表示，每个 token 都由一个 `d_model` 维向量表示。

------

## 4.2 Linear Projections in Code

These lines implement the projection equations:
 这三行实现的就是前面的投影公式：

```
Q = self.W_q(x)
K = self.W_k(x)
V = self.W_v(x)
```

Mathematically, they correspond to:
 数学上对应：
$$
Q = X W_Q,\qquad K = X W_K,\qquad V = X W_V
$$
In practice, PyTorch’s `nn.Linear` applies the transformation to the last dimension. So for each token representation, it produces a new projected vector.
 在实际中，PyTorch 的 `nn.Linear` 会对最后一个维度做线性变换，所以对于每个 token 表示，它都会得到一个新的投影向量。

------

## 4.3 Score Computation in Code

The line
 这一行：

```
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
```

computes all pairwise dot products between queries and keys.
 它在计算 query 和 key 的两两点积。

If `Q` has shape `(B, T, D)` and `K` has shape `(B, T, D)`, then `K.transpose(-2, -1)` has shape `(B, D, T)`, so the result has shape `(B, T, T)`.
 如果 `Q` 的形状是 `(B, T, D)`，`K` 的形状是 `(B, T, D)`，那么 `K.transpose(-2, -1)` 的形状就是 `(B, D, T)`，结果就是 `(B, T, T)`。

That `(T, T)` matrix for each batch tells us: for each token position, how strongly it attends to every token position in the sequence.
 对于每个 batch 来说，这个 `(T, T)` 的矩阵表示：每个 token 位置对序列中所有 token 位置的关注程度。

------

## 4.4 Softmax in Code

The line
 这一行：

```
attn_weights = F.softmax(scores, dim=-1)
```

normalizes along the last dimension, which means that for each query token, the weights across all key tokens sum to 1.
 它沿最后一个维度做归一化，也就是对每个 query token 来说，所有 key token 的权重之和为 1。

So each row of the attention matrix is a probability distribution over source positions.
 所以 attention 矩阵的每一行，都可以看作对源位置的一个概率分布。

------

## 4.5 Weighted Sum of Values in Code

The last line
 最后这一行：

```
output = torch.matmul(attn_weights, V)
```

takes the weighted sum of values. Since `attn_weights` has shape `(B, T, T)` and `V` has shape `(B, T, D)`, the output has shape `(B, T, D)`.
 它对 values 做加权求和。由于 `attn_weights` 的形状是 `(B, T, T)`，`V` 的形状是 `(B, T, D)`，所以输出的形状是 `(B, T, D)`。

This means every output token is a contextualized representation computed from the whole sequence.
 这意味着每个输出 token 都是结合了整个序列上下文之后得到的表示。

------

# 5. A Concrete Mini Example

Let us run a small example so the shapes become concrete.
 我们跑一个小例子，让这些 shape 更具体一些。

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
 预期输出：

```python
Input shape : torch.Size([2, 4, 8])
Output shape: torch.Size([2, 4, 8])
```

The shape stays the same because attention transforms each token representation into a new contextualized token representation of the same dimension.
 输出 shape 不变，因为 attention 的作用是把每个 token 表示转换成一个新的、带上下文的 token 表示，但维度还是原来的维度。

------

# 6. Why Masking Matters

In practice, attention often needs masking.
 在实际中，attention 往往需要 mask。

There are two major reasons.
 主要有两个原因。

First, some positions may be padding tokens and should not contribute information.
 第一，有些位置可能是 padding，不应该参与信息传递。

Second, in autoregressive language models like GPT, a token should not attend to future tokens.
 第二，在 GPT 这样的自回归语言模型里，当前 token 不能看到未来 token。

------

## 6.1 Causal Mask in PyTorch

Here is how to build a standard causal mask:
 下面是标准 causal mask 的写法：

```python
def causal_mask(seq_len: int) -> torch.Tensor:
    """
    Returns a lower-triangular mask of shape (1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)
```

Usage:
 使用方式：

```python
mask = causal_mask(T)
y = attn(x, mask=mask)
```

This lower-triangular mask ensures that position $i$ can only attend to positions up to $i$, not after it.
 这个下三角 mask 保证位置 $i$ 只能看到不超过 $i$ 的位置，而看不到未来位置。

------

# 7. Multi-Head Attention

Single-head attention already works, but Transformer uses multi-head attention because one similarity space is often not enough. Different relationships may require different projection subspaces.
 单头 attention 已经能工作，但 Transformer 使用 multi-head attention，因为一个相似度空间通常不够。不同类型的关系可能需要不同的投影子空间。

For example, one head may focus on syntactic structure, another on long-range reference, and another on local phrase composition.
 例如，一个 head 可能更关注句法结构，另一个更关注长距离指代，再另一个更关注局部短语组合。

Instead of forcing all these patterns into one Q/K/V space, we learn several smaller Q/K/V spaces in parallel.
 我们不把这些模式都塞进同一个 Q/K/V 空间里，而是并行学习多个更小的 Q/K/V 子空间。

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
 这就是完整的 multi-head 版本，但它在概念上和单头 attention 的流程是一样的。

------

## 7.2 What Changes from Single-Head to Multi-Head?

The main difference is that after projection, we split the representation dimension into multiple heads.
 和单头版本相比，最大的不同是：投影之后，我们会把表示维度切分成多个 head。

If `d_model = 512` and `num_heads = 8`, then each head works in a 64-dimensional space.
 如果 `d_model = 512` 且 `num_heads = 8`，那么每个 head 的维度就是 64。

Each head computes its own attention pattern independently.
 每个 head 都会独立计算自己的 attention pattern。

After that, all head outputs are concatenated and then passed through a final linear layer `W_o`.
 最后，再把所有 head 的输出拼接起来，并经过最终的线性层 `W_o`。

------

## 7.3 Why the Final Output Projection `W_o`?

A common question is: if we already concatenated all heads, why do we still need another linear layer?
 一个常见问题是：既然 head 都已经 concat 起来了，为什么还需要一个额外的线性层？

Because concatenation only stacks the information. It does not mix or re-coordinate the different subspaces.
 因为 concat 只是把信息堆在一起，并没有真正把不同子空间里的信息融合起来。

The final projection learns how to combine information across heads and map it back into the model space.
 最终的投影层会学习如何融合不同 head 的信息，并把结果映射回模型空间。

------

# 8. Connecting the Code Back to Theory

Now that we have seen the code, it is useful to reconnect it to the conceptual picture.
 看到代码之后，我们再把它和概念图景重新连起来，会更清楚。

In the code, the projections `W_q`, `W_k`, and `W_v` are the learned feature maps. They transform the raw token representation into three different latent spaces.
 在代码里，`W_q`、`W_k`、`W_v` 就是学习到的特征映射。它们把原始 token 表示变换到三个不同的隐空间里。

The operation `Q @ K^T` computes a similarity structure in the learned query-key space.
 `Q @ K^T` 这一步是在学习到的 query-key 空间中计算相似度结构。

The multiplication by `V` performs information aggregation in the learned value space.
 与 `V` 的乘法则是在学习到的 value 空间中进行信息聚合。

So the code reflects a very clean division of labor:
 所以，代码恰好对应了一个非常清晰的分工：

- Q/K define **where information should flow**
- V defines **what information should flow**
- Q/K 定义**信息该往哪里流动**
- V 定义**真正流动的信息是什么**

------

# 9. Attention as Kernel Regression

This brings us to the deeper statistical interpretation.
 这就把我们带到了更深的统计学解释。

Classical kernel regression has the form:
 经典核回归的形式是：
$$
f(x_i) = \sum_j K(x_i, x_j) f(x_j)
$$
This says that the estimate at point $x_i$ is a weighted combination of function values at other points, where the weights are determined by a kernel measuring similarity.
 它表示：点 $x_i$ 处的估计值，是其他点函数值的加权组合，而权重由核函数来决定相似度。

Attention has almost the same structure:
 Attention 几乎有完全一样的结构：
$$
\mathrm{Attn}(x_i) = \sum_j \alpha_{ij} v_j
$$
where the attention weights $\alpha_{ij}$ come from query-key similarity.
 其中注意力权重 $\alpha_{ij}$ 来自 query-key 相似度。

So we can interpret:
 所以我们可以把它解释为：

- $QK^\top$ defines a learned kernel
- $V$ provides the learned function values
- $QK^\top$ 定义了一个学习到的核函数
- $V$ 提供了学习到的函数值

This is why it is reasonable to say:
 这也是为什么可以说：

> Attention is kernel smoothing in a learned latent space.
>  Attention 本质上是在学习到的隐空间中进行核平滑。

------

# 10. Why This Interpretation Matters

This kernel view is not just philosophical. It helps explain why attention is so flexible.
 这个核方法视角并不只是哲学式解释，它能帮助说明 attention 为什么如此灵活。

In classical machine learning, the choice of kernel determines how similarity is measured. In attention, the model does not use a fixed kernel like RBF. Instead, it learns the feature maps that induce the kernel through $W_Q$ and $W_K$.
 在经典机器学习里，核函数决定了相似度怎么定义。而在 attention 里，模型并不是用固定的 RBF 核之类的函数，而是通过 $W_Q$ 和 $W_K$ 学习出诱导核函数的特征映射。

At the same time, $W_V$ learns what information should be passed around once similarity is established.
 同时，$W_V$ 还在学习：一旦相关性建立了，究竟应该传递什么信息。

So Transformer is jointly learning two things:
 所以 Transformer 实际上在同时学习两件事：

1. a similarity structure
2. a content representation

1）相似度结构
 2）内容表示

That is a very powerful combination.
 这是一个非常强大的组合。

------

# 11. Interview-Level Summary

If you need a concise but high-quality explanation in an interview, here is a strong version:
 如果你要在面试中给出一个简洁但高质量的解释，可以这样说：

Attention computes a weighted aggregation of value vectors, where the weights are determined by learned similarity between queries and keys. Queries encode what information a token is looking for, keys encode how tokens can be matched, and values encode the information being retrieved. From a statistical perspective, attention can be interpreted as a learned kernel regression in latent space.
 Attention 会对 value 向量做加权聚合，而权重由 query 和 key 之间学习到的相似度决定。Query 编码的是一个 token 在寻找什么信息，key 编码的是 token 如何被匹配到，value 编码的则是真正被检索和聚合的信息。从统计角度看，attention 可以被解释为在隐空间中的一个学习型核回归。

------

# 12. Final Takeaways

Attention is not merely a weighted average. It is a structured mechanism that separates relevance estimation from information transport. That separation is what makes it expressive, scalable, and suitable for modern large language models.
 Attention 并不只是一个简单的加权平均。它是一个有结构的机制，把相关性估计和信息传输显式分开。正是这种分离，让它具备强表达能力、良好扩展性，并适合现代大语言模型。

The PyTorch code makes this especially clear: the model first learns Q/K/V projections, then computes similarities, then normalizes them, and finally aggregates values. Every line of code reflects a conceptual decision.
 PyTorch 代码把这点展现得很清楚：模型先学习 Q/K/V 投影，然后计算相似度，再做归一化，最后聚合 values。代码的每一行背后都对应着一个概念决策。

The deeper kernel view gives an even stronger unification: attention learns both the kernel and the function representation at the same time.
 而更深的核方法视角则提供了一个更统一的理解：attention 同时在学习核函数和函数值表示。

That is why attention is not just useful engineering. It is also a beautiful statistical idea.
 这也是为什么 attention 不只是一个工程技巧，它本身也是一个非常漂亮的统计学思想。

------

# 13. Full PyTorch Demo

For completeness, here is a runnable example combining multi-head attention and a causal mask.
 为了完整起见，下面给出一个可运行的小例子，把 multi-head attention 和 causal mask 放在一起。

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
 这个例子已经非常接近 Transformer block 的核心了，除了还没加 residual connection、layer normalization 和 feed-forward network。

------

# 14. One-Sentence Summary

Attention is a learned retrieval mechanism in which Q and K define relevance, V defines content, and the whole computation can be viewed both as neural message passing and as kernel regression in latent space.
 Attention 是一种学习到的检索机制，其中 Q 和 K 定义相关性，V 定义内容，而整个计算既可以看成神经网络中的消息传递，也可以看成隐空间中的核回归。

# References

\- Hands-On Large Language Models  
\- Build a Large Language Model from Scratch  
\- StatQuest (YouTube)  
\- 李宏毅机器学习课程  
\- Deep-ML (花书相关资源)  
\- Datawhale Happy LLM  
\- https://zhuanlan.zhihu.com/p/626820422  
\- https://www.zhihu.com/question/298810062  
