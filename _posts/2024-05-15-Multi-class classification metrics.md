---
layout:       post
title:        "Multi-class classification metrics"
author:       "Nancycy"
header-style: text
catalog:      true
mathjax:      true
tags:
    - classification
    - metric
---

## General procedure

1. calculate TP/TN/FP/FN for each class
2. calculate accuracy
	- traditional accuracy
	- balanced accuracy
	- balanced accuracy weighted
1. calculate precision / recall / f1 that combine all classes
	- micro-averaging
	- macro-averaging

## Metrics for each class

Calculate TP/TN/FP/FN for each class. Suppose for each $x$, we want to predict its label $t$. Then for each prediction of $x$, we will investigate each class: does it belong to TP/TN/FP/FN.

Suppose we have 5 classes. If prediction is class 1, then the prediction for class 1 is positive; the prediction for class 2,3,4,5 is negative. Based on this, we can define TP/TN/FP/FN.

**Case 1**:
- prediction: class 1
- ground-truth: class 1
Result:
- $TP_{class_{i}} + = 1$ for $i=1$
- $TN_{class_{i}} + = 1$ for $i=2,3,4,5$

**Case 2**:
- prediction: class 1
- ground-truth: class 2
Result:
- $FP_{class_{i}} + = 1$ for $i=1$
- $FN_{class_{i}} + = 1$ for $i=2$
- $TN_{class_{i}} + = 1$ for $i=3,4,5$

### Variant: if prediction is a list (top k)

Suppose we have 5 classes, and we need to predict top 3 labels for each $x$. If prediction is class 1, 2, 3, then the prediction for class 1,2,3 is positive; the prediction for class 4, 5 is negative.

**Case 1**
- prediction: class 1, 2, 3
- ground-truth: class 1
Result:
- $TP_{class_{i}} + = 1$ for $i=1$
- $FP_{class_{i}} + = 1$ for $i=2, 3$
- $TN_{class_{i}} + = 1$ for $i=4,5$

**Case 2**
- prediction: class 1, 2, 3
- ground-truth: class 4
Result:
- $FP_{class_{i}} + = 1$ for $i=1,2,3$
- $FN_{class_{i}} + = 1$ for $i=4$
- $TN_{class_{i}} + = 1$ for $i=5$

## Accuracy

### Traditional Accuracy

If apply micro/macro concept, traditional accuracy is more like micro-level:

$$ \text { Accuracy }=\frac{\sum_{t \in T}\left(TP_t+TN_t\right)}{\sum_{t \in T}\left(TP_t+TN_t+FP_t+FN_t\right)} $$

### Balanced Accuracy

Traditional accuracy treats every sample equally, without considering the class distribution. So the metric actually depend more on the majority class. To treats every class equally, Balanced Accuracy is introduce. 

Each class has an equal weight in the final calculation of Balanced Accuracy, and each class is represented by its recall, regardless of their size. "Balanced" means every class has the same weight and the same importance.

$$ \text { BalancedAccuracy }=\frac{1}{|T|} \sum_{t \in T} \text { Recall }_t=\text { Recall }_M $$

### Balanced Accuracy Weighted

Balanced Accuracy Weighted is extended from Balanced Accuracy, where each recall multiplies its weight $w_t$ (the frequency of the class on the entire dataset). Let $W$ be the sum of all weights.

$$ \text { BalancedAccuracyWeighted }=\frac{1}{|T|*W} \sum_{t \in T} \frac{\text{ Recall }_t}{w_{k}}$$

> However, in the implementation of sklearn (Scikit-learn developers, 2020), the balanced accuracy does not count the scores for classes that did not receive predictions. So another **modified balanced** accuracy is:
>
> $$ \text { ModifiedBalancedAccuracy }_\gamma=\frac{1}{|X|} \sum_{t \in X} \text { Recall }_t $$
> 
> where $X$ denotes the set of all tags that have been predicted at least once.

## Micro averaging

Consider TP/FP/FN/TN globally, do not consider what class does it belong to. So micro-level mean summing up nominator and denominator separately before division. Next, we will use $t \in T$ to represent each class.

**Micro-level precision**:

$$ \text { Precision }_\mu=\frac{\sum_{t \in T} TP_t}{\sum_{t \in T}\left(TP_t+FP_t\right)} $$

**Micro-level recall**:

$$ \text { Recall }_\mu=\frac{\sum_{t \in T} TP_t}{\sum_{t \in T}\left(TP_t+FN_t\right)} $$

**Micro-level F-measure**:

$$ F-measure_\mu=2 \cdot \frac{\text { Precision }_\mu \cdot \text { Recall }_\mu}{\text { Precision }_\mu+\text { Recall }_\mu} $$

## Macro-averaging

At macro-level, calculate the metrics for each class t, and then calculate the unweighted or weighted mean.

**Macro-level precision**:

$$ \text { Precision }_M=\frac{1}{|T|}\sum_{t \in T}\frac{TP_t}{TP_t+FP_t} $$

**Macro-level recall**:

$$ \text { Precision }_M=\frac{1}{|T|}\sum_{t \in T}\frac{TP_t}{TP_t+FN_t} $$

**Macro-level F-measure**:

$$ F-measure_M=2 \cdot \frac{\text { Precision }_{M} \cdot \text { Recall }_M}{\text { Precision }_M+\text { Recall }_M} $$

## Scikit-Learn Implementation

### Average

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels, preds)
```

### Balanced Average

```python
from sklearn.metrics import balanced_accuracy_score
accuracy = balanced_accuracy_score(labels, preds)
```

### Balanced Average Weighted

```python
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight(class_weight='balanced', y=labels)
accuracy = balanced_accuracy_score(labels, preds)
```

### Micro-averaging precision/recall/f1

Calculate metrics globally by counting the total true positives, false negatives and false positives.
```python
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(labels, preds, average='micro')  
recall = recall_score(labels, preds, average='micro')  
f1 = f1_score(labels, preds, average='micro')
```

### Macro-averaging precision/recall/f1

Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
```python
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(labels, preds, average='macro')  
recall = recall_score(labels, preds, average='macro')  
f1 = f1_score(labels, preds, average='macro')
```

### Weighted-averaging precision/recall/f1

Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
```python
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(labels, preds, average='weighted')  
recall = recall_score(labels, preds, average='weighted')  
f1 = f1_score(labels, preds, average='weighted')
```

## Reference

1. (2021) Toward building recommender systems for the circular economy: Exploring the perils of the European Waste Catalogue
2. (2020) Metrics for Multi-class Classification