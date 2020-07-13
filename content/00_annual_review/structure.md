# Thesis Structure

```{admonition} WIP
:class: Tip
This content is work in progress.
```

````{margin}
```{note}
The title is not completely settled on. 
```
````
## Title
Fair Representations of Biased Data

## Table of Contents
1. Introduction
    1. Overview.
    2. Fair Machine Learning.
        - Definitions of fairness.
    3. Challenges.
2. Background
    1. Adversarial approaches.
    2. Distribution matching.
    3. Counterfactual Fairness.
    4. (possibly) Active Learning.
        1. types of uncertainty.
        2. model uncertainty.
        3. active learning.
3. Chapter 1
    1. Discovering Fair Representations in the Data Domain.
    2. Null-Sampling for Invariant and Interpretable Representations.
4. Chapter 2
    1. Imagined Examples for Fairness: Sampling Bias and Proxy Labels.
5. Chapter 3 (2 possibilities - need to determine which to pursue)
    - Fair Representations in the Data Domain and Counterfactuals.
    - Active Learning to reduce model uncertainty.
6. Conclusion

## Introduction
Discussion of research area, before focusing in on the research question.

## Background Work
Literature review of works to date. 
The literature review provided for year one was too broad.
This review will focus more specifically on learned representations.

## Chapter 1

The first chapter will be formed of 2 sections.

### Discovering Fair Representations in the Data Domain
This section will be formed of a journal submission based on expanding the [CVPR '19 paper](../09_appendix/dfritdd.md).
In this paper we decompose an image into a fair component ($\hat{x}$) and an unfair component ($\tilde{x}$).
We assume a decomposition of $\phi(x) = \phi(\hat{x}) + \phi(\tilde{x})$, where $\phi$ is a pre-trained feature map.
The expansion will be based on comparing HSIC (the independence measure used in the paper) to enforce independence vs 
adversarial learning and distribution matching.
In addition, results will be included experimenting with non-binary sensitive and class labels.

### Null-Sampling for Invariant and Interpretable Representations
This section will be based on our [ECCV '20](../09_appendix/nosinn.md) paper.
A natural successor to the CVPR '19 paper, we build on the previous work, but instead of a decomposition assumed as 
$\phi(x) = \phi(\hat{x}) + \phi(\tilde{x})$, we try to learn some decomposition function $d$, $x=d(\hat{x}, \tilde{x})$.
This work will be expended to include analysis of the interpretability of the representation in the data domain.  

## Chapter 2

This chapter will be based on the paper "Imagined Examples: Sampling Bias and Proxy Labels".
 
### Imagined Examples 
In previous works we have looked at amending bias that exists within the input space for a machine learning model.
However, that is not the only source of bias.
In addition to the input space, the recorded output can also exhibit bias.
An example of this behaviour is when the true behaviour is not (or cannot) be observed, and a proxy label is used instead.
In this work try to identify where bias is entering into a dataset and try to counteract the behaviour by generating
"imagined" (likely counterfactual) examples, and augment the original data with this, creating a balanced dataset.

## Chapter 3

There are currently two possible approaches for chapter 3.
One option is to link the work I have been doing with the Intervene project to my work on learned fair representations.
The alternative is to focus on some work I have recently been doing on Active Learning.

### Fair Representations in the Data Domain and Counterfactuals 
During my time working in the group I have been part of a team that has built the toolkit, Intervene.
Intervene builds causal models of data.
We have a hypothesis that fair representations of data are removing the causal effect of a sensitive attribute.
This would normally be very difficult to investigate as the representation is usually realised in an embedding space, so 
directly comparing two domains is challenging. 
However, in my work I have been building fair representations that occupy the same domain as the input space.
This allows the learnt representation and the original input to be directly compared.
In this work, I will compare the learnt causal model trained on the original data against the learnt causal model trained
on the fair representation of the data to test our hypothesis.

### Active Learning to reduce model uncertainty
In previous works we have considered the representation of the data.
One of the motivating factors why we often have to use biased on mis-representative data in the first place is that 
demographics change over time.
Our models being trained on historic data doesn't account for this demographic shift.
Active learning is a framework that introduces the most informative samples to a model over time, encouraging __efficient__
learning.

One of the disadvantages of active learning, however, is that it requires re-training a model from scratch at each timestep.
Given that modern machine learning models are becoming increasingly complex and training time can be in the region of several GPU years,
this is quite a burden.
In this work we will use a "replay memory" to augment the latest samples at a given timestep with a representation of the 
previously seen samples.
The aim is to produce results on-par with those of a model being re-trained on all of the previously seen data, but in fewer iterations.  