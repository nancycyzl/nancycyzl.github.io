---
layout:       post
title:        "Transformer - Multi-Query Attention (MQA) code from scratch"
author:       "Nancycy"
header-style: text
catalog:      true
mathjax:      true
tags:
    - LLMs
---

Compared with multi-head attention where each head has its own query, key, value, Multi-query Attention (MQA) heads only have individual queires while shaing the same set of key and value.

- query projection: `dim_in` -> `dim_k`
- key projection: `dim_in` -> `dk`  (`dk` = `dim_k` // n_head)
- value projection: `dim_in` -> `dv` (`dv` = `dim_v` // n_head)


## Pytorch implementation

```python
import torch
import torch.nn as nn
import math

class MultiQueryAttention(nn.Module):

    def __init__(self, dim_in, dim_k, dim_v, dim_o, n_head):
        super().__init__()

        assert dim_k % n_head == 0
        assert dim_v % n_head == 0

        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_o = dim_o
        self.n_head = n_head

        self.dk = self.dim_k // n_head
        self.dv = self.dim_v // n_head

        # define some projection layers
        self.linear_q = nn.Linear(self.dim_in, self.dim_k)
        self.linear_k = nn.Linear(self.dim_in, self.dk)    # here is dk instead of dim_k
        self.linear_v = nn.Linear(self.dim_in, self.dv)    # here is dv instead of dim_v
        self.linear_o = nn.Linear(self.dim_v, self.dim_o)


    def forward(self, x, mask=None):   # x dimension: (batch_size, seq_len, dim_in); mask shape (seq_len, seq_len)
        # get shape
        batch_size, seq_len, dim_in = x.shape
        assert dim_in == self.dim_in

        # get q, k, v
        q = self.linear_q(x).reshape(batch_size, seq_len, self.n_head, self.dk).transpose(1,2)      # (batch_size, n_head, seq_len, dk)
        k = self.linear_k(x).unsqueeze(1)      # (batch_size, 1, seq_len, dk)
        v = self.linear_v(x).unsqueeze(1)      # (batch_size, 1, seq_len, dv)

        # attention
        scale = 1 / math.sqrt(self.dk)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * scale    # (batch_size, n_head, seq_len, seq_len)

        # apply mask (1 means need to mask)
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores.masked_fill(mask==1, float('-inf'))

        # softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)   # (batch_size, n_head, seq_len, seq_len)
        
        # multiply by v
        output = torch.matmul(attention_probs, v)   # (batch_size, n_head, seq_len, dv)

        # concat and output linear
        output = output.transpose(1,2).reshape(batch_size, seq_len, self.dim_v)
        output = self.linear_o(output)   # # (batch_size, seq_len, dim_o)

        return output


def test():

    batch_size = 10
    seq_len = 100
    dim_in = 512
    dim_k = 128
    dim_v = 128
    dim_o = 64
    n_head = 8

    # create an input embedding
    input_emb = torch.randn(batch_size, seq_len, dim_in)

    # create a causal mask  (1 meeas need to be masked, upper triangle)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

    # test
    mqa = MultiQueryAttention(dim_in, dim_k, dim_v, dim_o, n_head)

    output = mqa(input_emb, mask=causal_mask)  # (batch_size, seq_len, dim_o) -> (10, 100, 64)

    print(output.shape)


if __name__ == "__main__":
    test()
```

