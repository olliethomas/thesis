# Thesis Structure

```{admonition} WIP
:class: Tip
This content is work in progress.
```

A two page outline of your thesis structure

## Title
Fair Representations in the Data Domain

## Introduction
Discussion of research area, before focussing in on the research question.

## Background Work
Literature review of works to date. 
The literature review provided for year one was too broad.
This review will focus more specifically on learned representations.

## Chapter 1
### Discovering Fair Representations in the Data Domain
This section will be formed of a journal submission based on expanding the [CVPR '19 paper](../09_appendix/dfritdd.md).
In this paper we decompose an image into a fair component ($\hat{x}$) and an unfair component ($\tilde{x}$).
We assume a decomposition of $\phi(x) = \phi(\hat{x}) + \phi(\tilde{x})$, where $\phi$ is a pre-trained feature map.
The expansion will be based on comparing HSIC to enforce independence vs adversarial learning and distribution matching.
In addition, additional results will be included experimenting with non-binary sensitive and class labels.

### Null-Sampling for Invariant and Interpretable Representations
This section will be based on our [ECCV '20](../09_appendix/nosinn.md) paper.
A natural successor to the CVPR '19 paper, we build on the previous work, but instead of a decomposition assumed as 
$\phi(x) = \phi(\hat{x}) + \phi(\tilde{x})$, we try to learn some decomposition function $d$, $x=d(\hat{x}, \tilde{x})$.
This work will be expended to include analysis of the interpretability of the representation in the data domain.  

## Chapter 2
### Imagined Examples 
In previous works we have looked at amending bias that exists within the input space for a machine learning model.
However, that is not the only source of bias.
In addition to the input space, the recorded output can also exhibit bias.
An example of this behaviour is when the true behaviour is not (or cannot) be observed, and a proxy label is used instead.
In this work try to identify where bias is entering into a dataset and try to counteract the behaviour by generating
"imagined" (likely counterfactual) examples, and augment the original data with this, creating a balanced dataset.

## Chapter 3
### Active Learning Work
In previous works we have considered the representation of the data.
One of the motivating factors why we often have to use biased on mis-representative data in the first place is that 
demographics change over time.
Our models being trained on historic data doesn't account for demographic shifts over time.
Active learning is a framework that introduces the most informative samples to a model over time.