---
layout:       post
title:        "Transformer - Group-Query Attention (GQA) code from scratch"
author:       "Nancycy"
header-style: text
catalog:      true
mathjax:      true
tags:
    - LLMs
---

Between multi-head attention and multi-query attention. For example, if `n_head` = 8, `n_group` = 2, then each 4 heads share one set of key and value.

- query projection: `dim_in` -> `dim_k`
- key projection: `dim_in` -> `dk` * `n_group`  (`dk` = `dim_k` // n_head)
- value projection: `dim_in` -> `dv` * `n_group` (`dv` = `dim_v` // n_head)

## Pytorch implementation


```python
import torch
import torch.nn as nn
import math

class GroupQueryAttention(nn.Module):

    def __init__(self, dim_in, dim_k, dim_v, dim_o, n_head, n_group):
        super().__init__()

        assert dim_k % n_head == 0
        assert dim_v % n_head == 0
        assert n_head % n_group == 0

        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_o = dim_o

        self.n_head = n_head    # query head
        self.n_group = n_group  # KV head
        self.group_size = self.n_head // self.n_group

        self.dk = self.dim_k // n_head
        self.dv = self.dim_v // n_head

        # define some projection layers
        self.linear_q = nn.Linear(self.dim_in, self.dim_k)
        self.linear_k = nn.Linear(self.dim_in, self.dk * self.n_group)    # here is dk instead of dim_k
        self.linear_v = nn.Linear(self.dim_in, self.dv * self.n_group)    # here is dv instead of dim_v
        self.linear_o = nn.Linear(self.dim_v, self.dim_o)


    def forward(self, x, mask=None):   # x dimension: (batch_size, seq_len, dim_in); mask shape (seq_len, seq_len)
        # get shape
        batch_size, seq_len, dim_in = x.shape
        assert dim_in == self.dim_in

        # get q, k, v
        q = self.linear_q(x).reshape(batch_size, seq_len, self.n_head, self.dk).transpose(1,2)      # (batch_size, n_head, seq_len, dk)
        k = self.linear_k(x).reshape(batch_size, seq_len, self.n_group, self.dk).transpose(1,2)     # (batch_size, seq_len, dk x n_group) -> (batch_size, n_group, seq_len, dk)
        v = self.linear_v(x).reshape(batch_size, seq_len, self.n_group, self.dv).transpose(1,2)     # (batch_size, seq_len, dv x n_group) -> (batch_size, n_group, seq_len, dv)

        
        # expand k and v: (batch_size, n_group, seq_len, dk) -> (batch_size, n_head, seq_len, dk)
        k = k.unsqueeze(2).expand(batch_size, self.n_group, self.group_size, seq_len, self.dk).reshape(batch_size, self.n_head, seq_len, self.dk)
        v = v.unsqueeze(2).expand(batch_size, self.n_group, self.group_size, seq_len, self.dv).reshape(batch_size, self.n_head, seq_len, self.dv)

        
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
    n_group = 2

    # create an input embedding
    input_emb = torch.randn(batch_size, seq_len, dim_in)

    # create a causal mask  (1 meeas need to be masked, upper triangle)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

    # test
    mqa = GroupQueryAttention(dim_in, dim_k, dim_v, dim_o, n_head, n_group)

    output = mqa(input_emb, mask=causal_mask)  # (batch_size, seq_len, dim_o) -> (10, 100, 64)

    print(output.shape)


if __name__ == "__main__":
    test()
```
