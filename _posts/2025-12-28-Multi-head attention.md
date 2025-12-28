---
layout:       post
title:        "Transformer - Multi-Head Attention (MHA) code from scratch"
author:       "Nancycy"
header-style: text
catalog:      true
mathjax:      true
tags:
    - LLMs
---

## Attention formular

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O
$$

$$
\mathrm{where} \quad \mathrm{head}_i = \mathrm{Attention}(Q W^Q_i, K W^K_i, V W^V_i)
$$



## Pytorch implementation

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, dim_o, n_head=8):
        super().__init__()

        assert dim_k % n_head == 0
        assert dim_v % n_head == 0

        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_o = dim_o
        self.n_head = n_head

        # define some layers
        self.linear_q = nn.Linear(self.dim_in, self.dim_k)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v)
        self.linear_o = nn.Linear(self.dim_v, self.dim_o)
        

    def forward(self, x, causal_mask=None, padding_mask=None):
        batch_size, seq_len, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.n_head
        dk = self.dim_k // nh   # d_q is the same
        dv = self.dim_v // nh

        # (bs, n, dim_in) 
        # -[linear_q]-> (bs, n, dim_k) 
        # -[reshape]-> (bs, n, nh, dk)
        # -[transpose] -> (bs, nh, n, dk) so that attention computes independely per head
        q = self.linear_q(x).reshape(batch_size, seq_len, nh, dk).transpose(1,2)
        k = self.linear_k(x).reshape(batch_size, seq_len, nh, dk).transpose(1,2)
        v = self.linear_v(x).reshape(batch_size, seq_len, nh, dv).transpose(1,2)

        # compute attention score (bs, nh, seq_len, seq_len)
        dk_tensor = torch.tensor(dk, dtype=torch.float32)
        attention_scores = torch.matmul(q, k.transpose(-1,-2)) / torch.sqrt(dk_tensor)

        # padding mask (bs, seq_len), if mask = 1, means need to mask
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)   # (bs, 1, 1, seq_len)
            attention_scores = attention_scores.masked_fill(padding_mask==1, float('-inf'))

        # causal mask (seq_len, seq_len), if mask = 1, means need to mask
        if causal_mask is not None:
            attention_scores = attention_scores.masked_fill(causal_mask==1, float('-inf'))

        # softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # multiply by v and combine all heads
        # result (bs, nh, seq_len, dv) -> (bs, seq_len, nh, dv) -> (bs, seq_len, dim_v) -> (bs, seq_len, dim_o)
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1,2).reshape(batch_size, seq_len, self.dim_v)
        output = self.linear_o(output)

        return output


def test_mha():
    batch_size = 24
    seq_len = 100
    dim_in = 1024
    dim_k = 512
    dim_v = 888
    dim_o = 2048
    n_head = 8

    # input embedding
    input_emb = torch.randn(batch_size, seq_len, dim_in)

    # create a mask: mask j>i (cannot see later tokens)
    # diagonal = 1: start above diagonal (keep diagonal j=i all 0)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    mha = MultiHeadAttention(dim_in, dim_k, dim_v, dim_o, n_head)
    output = mha(input_emb, causal_mask)

    print(output.shape)


if __name__ == "__main__":
    test_mha()
```

