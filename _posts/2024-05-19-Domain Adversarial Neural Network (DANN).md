---
layout:       post
title:        "Domain Adversarial Neural Network (DANN)"
author:       "Nancycy"
header-style: text
catalog:      true
mathjax:      true
tags:
    - neural network
    - algorithm
---

## Basics

A representation learning approach for domain adaptation. Trained on labeled data from the source domain and unlabeled data from the target domain.

Learn the features that have:
1. **discriminativeness**: discriminate for the main learning task on the source domain (eg. classification)
2. **domain-invariance**: indiscriminate between source and target domains

Key idea: embed domain adaptation into the process of learning representation, so that the final classification decisions are made based on features that are both discriminative and invariant to the change of domains.

The network contains a **label predictor** that predicts class labels during training and test time, and a **domain classifier** that discriminate between source and target domain during training. While the parameters of the classifers are optimized in order to minimize their error on the training set, the parameters of the underlying deep feature mapping are optimized in order to minimize the loss of the label classier and to maximize the loss of the domain classier.

In practice, the only non-standard component of the proposed architecture is a rather trivial **gradient reversal layer** that leaves the input unchanged during forward propagation and reverses the gradient by multiplying it by a negative scalar during the backpropagation.

[//]: # (![[Pasted image 20240319094309.png|700]])
<img src="/img/post2024/DANN.png" alt="image" width="700">

## Domain adaption (DA)

Learning a discriminative classifier or other predictor in the presence of a shift between training and test distributions is known as domain adaptation (DA). 

- **Unsupervised domain annotation (focus)**: target domain data are fully unlabeld.
- **Semi-supervised domain adaptation**: target domain data have few labeled samples.

### Formulation of unsupervised domain adaption

Consider a classification task where $X$ is the input space and $Y = {0, 1, ..., L-1}$ is the set of $L$ possible labels. There are two different distributions over $X \times Y$: source domain $\mathcal{D}\_{\mathrm{S}}$ and target domain $\mathcal{D}_{\mathrm{T}}$.

$$ S=\left\{\left(\mathrm{x}_i, y_i\right)\right\}_{i=1}^n \sim\left(\mathcal{D}_{\mathrm{S}}\right)^n $$

$$ T=\left\{\mathbf{x}_i\right\}_{i=n+1}^N \sim\left(\mathcal{D}_{\mathrm{T}}^X\right)^{n^{\prime}} $$

There are $n$ samples from $S$ and $n'$ from $T$, and $n+n'=N$. The goal is to build a classifier $\eta: X \rightarrow Y$ with a low target risk (to minimize misclassification rate) while having no information about the labels of $\mathcal{D}_{\mathrm{T}}$:

$$ R_{\mathcal{D}_{\mathrm{T}}}(\eta)=\operatorname{Pr}_{(\mathbf{x}, y) \sim \mathcal{D}_{\mathrm{T}}}(\eta(\mathbf{x}) \neq y) $$

### H-divergence
Reference: Ben-David et al., 2006, 2010; Kifer et al., 2004

Assume the hypothesis class $\mathcal{H}$ is a set of binary classifiers $\eta: X \rightarrow\{0,1\}$ to distinguish between source (label 0) and target (label 1) domains.

**Definition**:
Given two domain distributions $D_S^X$ and $D_T^X$ over $X$, and a hypothesis class $H$, the H-divergence between $D_S^X$ and $D_T^X$ is:

$$ d_H\left(D_S^X, D_T^X\right)=2 \sup _{\eta \in H}\left|\operatorname{Pr}_{x \sim D_S^X}[\eta(x)=1]-\operatorname{Pr}_{x \sim D_T^X}[\eta(x)=1]\right| $$
 
- **Hypothesis Class $H$:** This is a set of functions (or hypotheses) that map elements of $X$ to {0, 1}. 
- **H-divergence $d_H$​:** This is a measure of the difference between the two distributions $D_S^X$​ and $D_T^X$​ with respect to the hypothesis class $H$. Intuitively, it measures how distinguishable the two distributions are when using the best possible classifier from $H$.
- **Supremum $sup$⁡:** This refers to the "supreme" or the least upper bound. In this context, it is looking for the hypothesis $\eta$ within the class $H$ that maximizes the difference in the probability of predicting 1 between the source and target distributions.
- **Absolute value $\|...\|$:** By looking at the absolute difference in probabilities that samples from the source and target distributions are classified as belonging to a particular class (usually the positive class, denoted by 1), the H-divergence quantifies the disagreement between the two distributions. A larger difference means a larger divergence.

[//]: # (![[DANN.excalidraw#^frame=CDFiEOv8|800]])
<img src="/img/post2024/DANN_divergence.png" alt="image" width="800">

**Calculate H-divergence empirically**
Suppose we have two samples $S \sim\left(\mathcal{D}\_{\mathrm{S}}^X\right)^n$ and $S \sim\left(\mathcal{T}_{\mathrm{T}}^X\right)^n$, the empirical H-divergence can be calculated as:

$$ \hat{d}_{\mathcal{H}}(S, T)=2\left(1-\min _{\eta \in \mathcal{H}}\left[\frac{1}{n} \sum_{i=1}^n I\left[\eta\left(\mathbf{x}_i\right)=0\right]+\frac{1}{n^{\prime}} \sum_{i=n+1}^N I\left[\eta\left(\mathbf{x}_i\right)=1\right]\right]\right) $$

where $I[a]$ is the indicator function which is 1 if $a$ is true, and 0 otherwise.

- **minimization term** : measures the misclassification rate of the hypothesis $\eta$ when it tries to discriminate between the source and target samples. The hypothesis that minimizes this rate is considered the best at differentiating between the two domains under the hypothesis class H.
- **2(1−minimization term)**: this outer expression converts the misclassification rate into an estimate of divergence. By subtracting the minimization expression from 1, you get a measure of how well the best hypothesis $\eta$ can fail to distinguish between the two domains, and then scaling by 2 aligns this empirical measure with the theoretical range of H-divergence.

### Proxy distance
Even if it is hard to computer $\hat{d}_{\mathcal{H}}(S, T)$, we can approximate it by running a learning algorithm to discriminate between source and target examples. A new dataset is constructed: 

$$ U=\left\{\left(\mathbf{x}_i, 0\right)\right\}_{i=1}^n \cup\left\{\left(\mathbf{x}_i, 1\right)\right\}_{i=n+1}^N $$

where examples from the source samples are labeled 0 and examples of the target samples are labeled 1. Hence, the H-divergence is then approximated by

$$ \hat{d}_{\mathcal{A}}=2(1-2 \epsilon) $$

where $\epsilon$ is the generalization error on the problem of discriminating between source and target examples. This $\hat{d}_{\mathcal{A}}$ value is called the **Proxy $\mathcal{A}$-distance** (PAD).

**$\mathcal{A}$-distance** is defined as:

$$ d_{\mathcal{A}}\left(\mathcal{D}_{\mathrm{S}}^X, \mathcal{D}_{\mathrm{T}}^X\right)=2 \sup _{A \in \mathcal{A}}\left|\operatorname{Pr}_{\mathcal{D}_{\mathrm{S}}^X}(A)-\operatorname{Pr}_{\mathcal{D}_{\mathrm{T}}^X}(A)\right| $$

where $\mathcal{A}$ is a subset of $X$. 

**$\mathcal{A}$-distance and $\mathcal{H}$-divergence**

By choose $$\mathcal{A}=\left\{A_\eta \mid \eta \in \mathcal{H}\right\}$$, with $A_\eta$ the set represented by the characteristic function $\eta$, the two are identical.

## Domain-adversarial Neural Networks

### A shallow NN
Consider a NN with a single hidden layer. 
Input space: $m$-dimensional real vectors. $X=\mathbb{R}^m$
Hidden layer $G_f$: map an example to a new $D$-dimensional representation. $G_f: X \rightarrow \mathbb{R}^D$

$$ G_f(\mathbf{x} ; \mathbf{W}, \mathbf{b})=\operatorname{sigm}(\mathbf{W} \mathbf{x}+\mathbf{b}) $$

Prediction layer $G_y$: map hidden representation to a output (0/1). $G_y: \mathbb{R}^D \rightarrow[0,1]^L$

$$ G_y\left(G_f(\mathbf{x}) ; \mathbf{V}, \mathbf{c}\right)=\operatorname{softmax}\left(\mathbf{V} G_f(\mathbf{x})+\mathbf{c}\right) $$

Given a source example $(\mathbf{x}_i, y_i)$, the classification loss is the negative log-probability of the correct label:

$$ \mathcal{L}_y\left(G_y\left(G_f\left(\mathbf{x}_i\right)\right), y_i\right)=\log \frac{1}{G_y\left(G_f(\mathbf{x})\right)_{y_i}} $$

Training the network leads to the optimization problem on the source domain:

$$ \min _{\mathbf{W}, \mathbf{b}, \mathbf{V}, \mathbf{c}}\left[\frac{1}{n} \sum_{i=1}^n \mathcal{L}_y^i(\mathbf{W}, \mathbf{b}, \mathbf{V}, \mathbf{c})+\lambda \cdot R(\mathbf{W}, \mathbf{b})\right] $$

### DANN method
The heart of the approach is to design a domain regularizer directly derived from the $H$-divergence. Let the output of hidden layer $G_f(\cdot)$ as the internal representation.

Source sample representation:

$$ S\left(G_f\right)=\left\{G_f(\mathrm{x}) \mid \mathrm{x} \in S\right\} $$

Target sample representation:

$$ T\left(G_f\right)=\left\{G_f(\mathrm{x}) \mid \mathrm{x} \in T\right\} $$

Therefore, the empirical $H$-divergence between $S\left(G_f\right)$ and $T\left(G_f\right)$ is:

$$ \hat{d}_{\mathcal{H}}\left(S\left(G_f\right), T\left(G_f\right)\right)=2\left(1-\min _{\eta \in \mathcal{H}}\left[\frac{1}{n} \sum_{i=1}^n I\left[\eta\left(G_f\left(\mathbf{x}_i\right)\right)=0\right]+\frac{1}{n^{\prime}} \sum_{i=n+1}^N I\left[\eta\left(G_f\left(\mathbf{x}_i\right)\right)=1\right]\right]\right) $$

How to calculate the "min" part? Use a domain classification layer $G_d$ that learns a logistic regressor $G_d: \mathbb{R}^D \rightarrow[0,1]$:

$$ G_d\left(G_f(\mathbf{x}) ; \mathbf{u}, z\right)=\operatorname{sigm}\left(\mathbf{u}^{\top} G_f(\mathbf{x})+z\right) $$

The loss is:

$$ \mathcal{L}_d\left(G_d\left(G_f\left(\mathbf{x}_i\right)\right), d_i\right)=d_i \log \frac{1}{G_d\left(G_f\left(\mathbf{x}_i\right)\right)}+\left(1-d_i\right) \log \frac{1}{1-G_d\left(G_f\left(\mathbf{x}_i\right)\right)} $$

where $d_i$ denotes the binary domain label for the $i$-th example.

Therefore, the complete optimization objective integrating $\mathcal{L}_d$ is:

$$ E(\mathbf{W}, \mathbf{V}, \mathbf{b}, \mathbf{c}, \mathbf{u}, z)=\frac{1}{n} \sum_{i=1}^n \mathcal{L}_y^i(\mathbf{W}, \mathbf{b}, \mathbf{V}, \mathbf{c})-\lambda\left(\frac{1}{n} \sum_{i=1}^n \mathcal{L}_d^i(\mathbf{W}, \mathbf{b}, \mathbf{u}, z)+\frac{1}{n^{\prime}} \sum_{i=n+1}^N \mathcal{L}_d^i(\mathbf{W}, \mathbf{b}, \mathbf{u}, z)\right) $$

The parameters are saddle point given by:

$$ \begin{aligned} (\hat{\mathbf{W}}, \hat{\mathbf{V}}, \hat{\mathbf{b}}, \hat{\mathbf{c}}) & =\underset{\mathbf{W}, \mathbf{V}, \mathbf{b}, \mathbf{c}}{\operatorname{argmin}} E(\mathbf{W}, \mathbf{V}, \mathbf{b}, \mathbf{c}, \hat{\mathbf{u}}, \hat{z}) \\ (\hat{\mathbf{u}}, \hat{z}) & =\underset{\mathbf{u}, z}{\operatorname{argmax}} E(\hat{\mathbf{W}}, \hat{\mathbf{V}}, \hat{\mathbf{b}}, \hat{\mathbf{c}}, \mathbf{u}, z) \end{aligned} $$

The method tackles this problem with a simple stochastic gradient procedure, in which updates are made in the opposite direction of the gradient for the minimizing parameters, and in the direction of the gradient for the maximizing parameters.

[//]: # (![[Pasted image 20240317224330.png|800]])
<img src="/img/post2024/DANN_algo.png" alt="image" width="800">

## Implementation

The whole structure contains 3 parts:
- encoder / feature extractor: to extract latent features of input samples
- classifier / label predictor: to predict the class label (the main problem we are looking at)
- discriminator / domain classifier: to predict the domain label

**Reverse layer**
1. **`ctx`**: This is a context object that can be used to stash information for backward computation. You don't provide it directly when calling the function. It's handled by PyTorch's autograd system.
2. **`x`**: This is the input tensor that you want to apply the operation to.
3. **`alpha`**: This is an additional parameter that scales the gradient during the backward pass.

```python
from torch.autograd import Function

class ReverseLayerF(Function):  
  
    @staticmethod  
    def forward(ctx, x, alpha):  
        ctx.alpha = alpha  
  
        return x.view_as(x)  
  
    @staticmethod  
    def backward(ctx, grad_output):  
        output = grad_output.neg() * ctx.alpha  
  
        return output, None
```

**Apply reverse layer in discriminator**
```python
class Discriminator(nn.Module):  
    def __init__(self, hidden_size):  
        super(Discriminator, self).__init__()  
  
        # domain classification layer  
        self.discriminator = nn.Sequential(  
            nn.Linear(in_features=hidden_size, out_features=int(hidden_size/2)),  
            nn.ReLU(),  
            nn.Linear(in_features=int(hidden_size/2), out_features=1)  
        )  
  
    def forward(self, input_feature, alpha):  
        reversed_input = ReverseLayerF.apply(input_feature, alpha)  
        x = self.discriminator(reversed_input)  
        return x
```

**During training, apply discriminator**
```python
# discriminator output  
alpha = 2. / (1. + np.exp(-10 * p)) - 1  
domain_pred_logits = discriminator(combined_features, alpha).squeeze()  
return class_pred_logits, domain_pred_logits, source_labels, target_labels
```

[//]: # (![[Excalidraw/DANN.excalidraw.md#^frame=_Wp-m2rcqGNA7tLo6GewR|500]])
<img src="/img/post2024/DANN_alpha.png" alt="image" width="400">

**During training, combine both loss**
```python
# classifier loss  
class_loss = classifer_criterion(class_pred_logits, source_labels)  
  
# discrinimator loss  
domain_source_labels = torch.zeros(source_labels.shape[0]).float()  
domain_target_labels = torch.ones(target_labels.shape[0]).float()  
domain_combined_labels = torch.cat((domain_source_labels, domain_target_labels), 0).to(args.device)  
domain_loss = discriminator_criterion(domain_pred_logits, domain_combined_labels)  
  
# combine loss and backward  
combined_loss = class_loss + domain_loss  
combined_loss.backward()  
optimizer.step()
```

