---
layout:       post
title:        "Complexity (NP related)"
author:       "Nancycy"
header-style: text
catalog:      true
tags:
    - Complexity
---

Reference:
1. https://www.baeldung.com/cs/p-np-np-complete-np-hard
2. https://blog.csdn.net/huang1024rui/article/details/49154507

### Big-O notation for complexity

- $O(1)$ - constant time
- $O(log_2(n))$ - logarithmic time
- $O(n)$ - linear time
- $O(n^2)$ - quadratic time
- $O(n^k)$ - polynomial time
- $O(k^n)$ - exponential time
- $O(n!)$ - factorial time

## P problem

P stands for "polynomial". The problem can be solved in polynomial time. Generally, the complexity can be written as 
$$T(n)=O(C*n^k)$$
where $C>0$ and $k>0$, $C$ and $k$ are constant, $n$ is input size.  Usually, $k$ is expected to be less than $n$.

## NP problem

NP stands for "non-deterministic polynomial". The problem cannot be solved in polynomial time, but can be verified / certified in polynomial time. The complexity is expected to be
$$T(n)=O(C_{1}*k^{C_{2*n}})$$
## NP-complete problem

They all belong to NP problem, but they hold another characteristic: completeness. For any NP problem that is complete, there exists a polynomial-time algorithm that can transform it into other NP-complete problem. This transformation is called reduction.

约化/归约 reduction:
问题A可以约化为问题B，指可以用问题B的解法解决问题A。NPC问题难度>= NP问题。其他所有的NP问题都可以约化为NPC问题。比如一元一次方程可以约化为一元二次方程。

NPC问题的条件
1. 是一个NP问题
2. 所有NP问题都可以约化到它

证明一个问题是NPC问题
1. 证明是一个NP问题
2. 证明一个已知的NPC问题能约化到它（由约化的传递性）

## NP-hard problem

The solution cannot be verified in polynomial time. They are at least as hard as any other problem in NP.

![[Pasted image 20240327223757.png|397]]