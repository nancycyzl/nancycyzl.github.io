---
layout:       post
title:        "Visualizing data using t-SNE"
author:       "Nancycy"
header-style: text
catalog:      true
mathjax:      true
tags:
    - algorithm
    - visualization
---

<img src="/img/post2024/Pasted image 20240625135338.png" alt="image" width="737">

## Idea

### Similarity Computation in High-dimensional Space
 
t-SNE starts by converting the Euclidean distances between points in the high-dimensional space into conditional probabilities that represent similarities. The similarity of datapoint $x_j$ to datapoint $x_{i}$​ is the probability that $x_{i}$​ would pick $x_{j}$​ as its neighbor if neighbors were picked in proportion to their probability density under a **Gaussian** centered at $x_{i}$​. This probability is symmetrized by averaging it with the probability of $x_{j}$​ picking $x_{i}$​ as a neighbor.

### Similarity Computation in Low-dimensional Space

In the low-dimensional space, t-SNE calculates a similar probability but uses a **t-distribution** (which has heavier tails than a Gaussian distribution) to measure distances between points. The heavier tails of the t-distribution help to alleviate the **crowding problem** where many points tend to collapse together when using other dimensionality reduction techniques.

### Minimizing the Kullback-Leibler Divergence

The main objective of t-SNE is to minimize the divergence between these two probability distributions: the high-dimensional **Gaussian distribution** and the low-dimensional **t-distribution**. This minimization is done using **gradient descent**. Essentially, t-SNE adjusts the locations of points in the low-dimensional space so that the probability distribution as closely as possible matches the distribution in the high-dimensional space.

## Implementation

### Key Parameters in t-SNE:

- **n_components**: The number of dimensions in which to embed the data. For visualization, this is usually 2 or 3.
- **perplexity**: A key parameter in t-SNE, often viewed as a guess about the number of close neighbors each point has. Typical values are between 5 and 50.
- **n_iter**: The number of iterations for optimization. Usually, a higher number gives more accurate results but takes longer.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Sample data - let's assume X is your dataset
# X should be a 2D array of shape (n_samples, n_features)

# Create a t-SNE object with desired parameters
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

# Fit and transform the data
tsne_results = tsne.fit_transform(X)

# Plotting
plt.figure(figsize=(12,8))
plt.scatter(tsne_results[:,0], tsne_results[:,1])
plt.show()

```