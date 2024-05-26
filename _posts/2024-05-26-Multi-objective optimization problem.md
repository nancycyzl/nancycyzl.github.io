---
layout:       post
title:        "Multi-objective optimization problem"
author:       "Nancycy"
header-style: text
catalog:      true
mathjax:      true
tags:
    - optimization
---

## Definition

- Involve more than one objective function that are to be minimized or maximized
- Answer is set of solutions that define the best trade-off between competing objectives

### Dominance Test

Solution $x_1$ dominates $x_2$ if
- solution $x_1$ is no worse (at least same good) than $x_2$ in all objectives
- solution $x_1$ is strictly better than $x_2$ in at least one objective

<img src="/img/post2024/MOOP_dominance.png" alt="image" width="400">

[//]: # (![[Excalidraw/Multi-objective optimization problem.excalidraw.md#^frame=H0qDxMW3mAFvSGYS9ZpQ3]])

**Non-dominated solution set** <br>
a set of all the solutions that are not dominated by any member of the solution set

### Pareto optimal solution

**Pareto optimal set**: the non-dominated set of the entire feasible decision space <br>
**Pareto optimal front**: the boundary defined by the set of all points mapped from the Pareto optimal set

### Goals in Multi-objective optimization

1. find set of solutions as close as possible to Pareto optimal front
2. find a set of solutions as diverse as possible

<img src="/img/post2024/Pasted image 20240326235342.png" alt="image" width="400">


## Classical methods

Original problem (m objectives):

$$ \begin{array}{ll} \min / \max & f_m(\boldsymbol{x}), \quad m=1,2, \cdots, M \\ \text { subject to } & g_j(\boldsymbol{x}) \geq 0, \quad j=1,2, \cdots, J \\ & h_k(\boldsymbol{x})=0, \quad k=1,2, \cdots, K \\ & x_{i}^{(L)} \leq x_i \leq x_{i}^{(U)}, \quad i=1,2, \cdots, n \end{array} $$

### Weighted sum method

Combine all objectives into a single objective using weighted sum. (scalarization)

**Modified objective function**:

$$ F(\boldsymbol{x})=\sum_{m=1}^M w_m f_m(\boldsymbol{x}) $$

**Advantage**:
- simple

**Disadvantage**:
- difficult to set the weight vectors to obtain a Pareto-optimal solution in a desired region in the objective space
- cannot find certain Pareto-optimal solutions in the case of a nonconvex objective space

<img src="/img/post2024/MOOP_convex.png" alt="image" width="600">


### $\epsilon$-constraint method

Keep just one objective and restricting the rest of objectives within user-specific values

**Modified problem**:

$$ \begin{array}{lll} \operatorname{minimize} & f_\mu(\boldsymbol{x}), \\ \text { subject to } & f_m(\boldsymbol{x}) \leq \varepsilon_m, & m=1,2, \cdots, M \text { and } m \neq \mu \\ & g_j(\boldsymbol{x}) \geq 0, & j=1,2, \cdots, J \\ & h_k(\boldsymbol{x})=0, & k=1,2, \cdots, K \\ & x_i^{(L)} \leq x_i \leq x_i^{(U)}, \quad i=1,2, \cdots, n \end{array} $$

<img src="/img/post2024/MOOP_eposilon.png" alt="image" width="700">


**Advantage**
- applicable to either convex or non-convex problems

**Disadvantage**
- the $\epsilon$ vector need to be chosen carefully so that it is within the minimum or maximum values of the individual objective function

[//]: # (**Example**)

[//]: # ([[Epsilon-constraint method]])

### Weighed metric method

Combine multiple objectives using the weighted distance metric of any solution from the idea solution $z^*$

**Modified problem**:

$$ \begin{array}{ll} \operatorname{minimize} & l_{\mathbf{p}}(\boldsymbol{x})=\left(\sum_{m=1}^M w_m\left|f_m(\boldsymbol{x})-z_m^*\right|^p\right)^{1 / p} \\ \text { subject to } & g_j(\boldsymbol{x}) \geq 0, \quad j=1,2, \cdots, J \\ & h_k(\boldsymbol{x})=0, \quad k=1,2, \cdots, K \\ & x_i^{(L)} \leq x_i \leq x_i^{(U)}, \quad i=1,2, \cdots, n \end{array} $$

<img src="/img/post2024/MOOP_weighted.png" alt="image" width="800">



**Advantage**
- weighted Tchebycheff metric guarantees finding all Pareto-optimal solution with idea solution $z^*$

**Disadvantage**
- require knowledge of minimum and maximum objective values
- require $z^*$ (can be found by independently optimizing each objective functions)
- for small $p$, not all Pareto-optimal solutions are obtained
- As $p$ increases, the problem becomes non-differentiable

## Multi-objective Genetic algorithms (MOEAs)

Advantage over classical methods
- classical: operate on one candidate solution
- GA: operate on a set of candidate solutions

Types of MOEA based on the use of elitist
- non-elitist MOEAs
- elitist MOEAs

<img src="/img/post2024/Pasted image 20240327003643.png" alt="image" width="400">


## Reference

1. https://codemonk.in/blog/a-gentle-introduction-to-multi-objective-optimization/
2. https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf
3. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6077796











