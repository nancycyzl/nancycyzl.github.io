---
layout:       post
title:        "Bi-encoder and Cross-encoder"
author:       "Nancycy"
header-style: text
catalog:      true
mathjax:      true
tags:
    - AI
    - encoding
    - RAG
---

Consider a query and a document, we want to compute the relatedness (e.g., similarity) between them.

## Bi-encoder

Use two separate encoders to encode query and document into $v_q$ and $v_d$, then use cosine similarity or dot product to compute the score.

### Examples

- **DPR (Dense Passage Retrieval)** - 2020
    - Developed by Facebook AI Research, DPR is used for open-domain question answering by encoding passages and queries independently.
- **SBERT (Sentence-BERT)** - 2019
    - Introduced by Reimers and Gurevych, SBERT uses BERT to generate sentence embeddings for semantic textual similarity tasks.
- **ColBERT (Contextualized Late Interaction over BERT)** - 2020
    - Developed by Khattab and Zaharia, ColBERT uses a bi-encoder architecture with late interaction to enhance passage retrieval tasks.
- **TAS-B (Twin-Answer Scorer for Bi-Encoder)** - 2021
    - A model that extends bi-encoders for scoring by using cross-attention to refine representations after initial encoding.
- **ANCE (Approximate Nearest Neighbor Negative Contrastive Learning)** - 2021
    - This model improves bi-encoder training using approximate nearest neighbor search for efficient negative sampling.

### Bi-encoder with MLP

Instead of simply use cosine similarity or dot product, we can use MLP on $[v_q, v_d]$ as the scoring function. With MLP, it can capture more complicated relationships.

**Benefits**:
- **Enhanced Expressiveness**: An MLP can capture more complex relationships between the vector representations compared to simple similarity measures.
- **Flexibility**: The MLP can be tailored to different tasks and can learn task-specific scoring functions.
- **Non-linearity**: Adding non-linear activation functions in the MLP can model non-linear interactions between the encoded vectors.

## Cross-encoder

Concatenate the query and the document, and use a single encoder + classification layer to output the score.

### Examples

1. **BERT (Bidirectional Encoder Representations from Transformers)** - 2018
    - While BERT itself is not a cross-encoder, it has been used as a foundation for many cross-encoder applications by concatenating input pairs for joint encoding.
2. **Cross-Encoder for Reranking** - 2019
    - Nogueira and Cho used BERT as a cross-encoder for passage reranking, demonstrating significant improvements in retrieval tasks.
3. **RoBERTa (Robustly optimized BERT approach)** - 2019
    - By Liu et al., RoBERTa is an optimized version of BERT used extensively as a cross-encoder in various tasks.
4. **T5 (Text-To-Text Transfer Transformer)** - 2020
    - Developed by Raffel et al., T5 can be used as a cross-encoder by framing tasks as text-to-text transformations, encoding input pairs jointly.
5. **DeBERTa (Decoding-enhanced BERT with Disentangled Attention)** - 2021
    - Introduced by He et al., DeBERTa uses enhanced self-attention mechanisms and is employed as a cross-encoder in several NLP tasks.

## How to choose

Generally, Cross-encoders have higher performance because they can capture fine-grained relationships. Bi-encoder obtain the vectors separately, so it cannot capture the interactions between the two inputs.

But Cross-encoders are not so efficient than Bi-encoders. With Bi-encoders, we can obtain a pool of vectors (all the candidate documents) first, and then just compute the similarity score if there is a new query coming in. But with cross-encoder, we need to run the encoding for every (query, document) pair, which is not efficient.

In summary:

- **Bi-Encoder**: Encodes inputs independently, suitable for scalable and efficient retrieval tasks, but with limited interaction modeling.
- **Cross-Encoder**: Encodes inputs jointly, capturing rich interactions at the cost of computational efficiency, suitable for tasks requiring detailed relationship understanding.