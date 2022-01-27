# Thesis Structure

## Title
Fair Representations of Biased Data

## Table of Contents
1. Introduction
   - Research Question: Can we learn to produce a representation of data that makes downstream tasks **fair**?
   - What are the potential benefits of doing this?
2. Background: Literature Review
    1. Fair Machine Learning.
        - Definitions of fairness.
    2. Adversarial approaches.
    3. Distribution matching.
    4. Causality.
       - Counterfactual Fairness.
    5. Challenges.
3. **Chapter 1**
   Without access to additional data, what changes can be made to make the data itself _fair_?   

   Papers
    1. Discovering Fair Representations in the Data Domain.
    2. Null-Sampling for Invariant and Interpretable Representations.
4. **Chapter 2**
   Which candidates appear susceptible to unfair treatment?

   Papers
    1. An Algorithmic Framework for Positive Action 
5. **Chapter 3**
   How certain is a model that the representation learned is fair?   

   Papers
   1. Fair Uncertainty Quantification of Learned Representation (**WIP**)
6. Conclusion

## Introduction
This chapter opens with my research question: Is it possible to learn to produce representations of data that provide enough information
for a downstream classification model to have utility, but also obfuscate protected characteristics?
In addition, I introduce the overall theme of the work, providing mechanisms to give feedback to a designer.
Different types of feedback are discussed in each chapter:
  - Chapter 1 consists of providing feedback about what needs to change about the data to become "fair".
  - Chapter 2 looks at providing feedback at deployment/inference time to ask which individuals are most at risk of an unfair decision.
  - Chapter 3 looks at methods which add a quantification of certainty around the "fairness" of a learned representation.

This is followed by a discussion of the topic, with a particular focus on the benefits of this area of research, and the challenges.

## Background Work
This chapter consists of a literature review of works to date. 
The topic of fair machine learning is very broad, so there is a particular focus on learned representations.

The structure is:
1. What is Fair Machine Learning?
   - Definitions of fairness.
      - Group
      - Individual
      - Counterfactual
2. A history of Fair Representation Learning.
3. Adversarial approaches.
4. Distribution matching.
5. Causality.
   - Counterfactual Fairness.
6. Challenges.

## Chapter 1
````{margin}
```{note}
The aim is to submit this chapter to a journal at the same time as submitting the thesis. 
```
````
The first chapter ties together work completed in two papers:
- [Discovering Fair Representations in the Data Domain](../../09_appendix/publications/dfritdd.md) (DFR)
   - Published at CVPR 2019[^cvpr2019]
- [Null-sampling for Invariant and Fair Representations](../../09_appendix/publications/nosinn.md) (NIFR)
   - Published at ECCV 2020[^eccv2020]

In both works, the aim is to produce a modified version of the input so that a downstream model produces a _fair_-er 
prediction when trained using this version, as opposed to the input.
The main difference between the works is the assumptions about how the data can be decomposed. 
In DFR, it is assumed that an input consists of the "fair" version of the image with an "unfair" mask added to it. 
e.g. $\textrm{input} = \textrm{fair input} + \textrm{unfair mask}$.
However, in NIFR, we assume the input is the result of an unknown function, $d(\cdot, \cdot)$ e.g. 
$\textrm{input} = d(\textrm{fair input}, \textrm{unfair mask})$.

I present both papers together in this chapter, with extensions to both from their conference versions.

### Discovering Fair Representations in the Data Domain
In this paper we decompose an image into a fair component ($\hat{x}$) and an unfair component ($\tilde{x}$).
We assume a decomposition of $\phi(x) = \phi(\hat{x}) + \phi(\tilde{x})$, where $\phi$ is a pre-trained feature map.

For this paper I was responsible for:
   - writing code
   - writing sections of the text
   - experiments on the tabular data

This chapter will extend the conference paper by comparing HSIC (the independence measure used in the paper) to enforce 
independence vs adversarial learning and distribution matching.
In addition, results will be included experimenting with non-binary sensitive and class labels.

### Null-Sampling for Invariant and Interpretable Representations
A natural successor to the CVPR '19 paper, we build on the previous work, but instead of a decomposition assumed as 
$\phi(x) = \phi(\hat{x}) + \phi(\tilde{x})$, we try to learn some decomposition function $d$, $x=d(\hat{x}, \tilde{x})$.
This work will be expended to include analysis of the interpretability of the representation in the data domain.  

For this paper I was responsible for:
  - writing code
  - running experiments
  - writing sections of the text
  - designing the evaluation
  - proposing methods to overcome challenges (e.g. using an AE as a pre-processor for the tabular data.)

This chapter will extend the conference paper by providing more analysis of the results for interpretability.

## Chapter 2

This chapter consists of the paper ["An Algorithmic Framework for Positive Action"](../../03_positive_action/intro.md), 
submitted to Data Mining and Knowledge Discovery Special issue on Bias and Fairness in AI.[^dami2021]
A shorter version is also under review at the inaugural ACM conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO).

In this work I try to determine, at inference time, individuals who are at risk of receiving a biased decision.
I do this using a relaxed form of Counterfactual Fairness. 
In this definition of fairness, I ask if the decision would have been affected if the individual had a different value
for their protected attribute (i.e. they had been born of a different race or gender). 
To 'properly' perform counterfactual modeling requires access to a data-generation procedure called a Structural Causal Model.
As these are difficult to obtain, I make use of the excellent performance that models such as StarGAN {cite}`ChoChoKimHaKimCho18`
and Cycle-Gan {cite}`ZhuParIsoEfr17` achieve in unsupervised image-to-image domain translation, where domains include, 
gender, hair colour, race, or other visible differences.

In this work I make use of fair representations which, in the language of the above models, encourages invariance to a specific domain
using additional data as opposed to inferring this from a large amount of data.
Because of this additional data, fewer samples are required to produce a similar result.

I use these domain translation results as our counterfactual examples for interrogating an existing model.
If the outcomes are different across counterfactuals, then we provide a strategy to handle this using the notion
of Positive Action.

In this work I was responsible for:
  - Writing code
  - Running experiments
  - Writing sections of the text


## Chapter 3

In this chapter I extend the work of chapter 1 by providing uncertainty estimates of the model used to provide a fair and interpretable representation.
This is achieved by using a Monte-Carlo Dropout approach, providing multiple hypotheses of a fair representation for each input.
The majority of this chapter is based on analysing where the representations are in consensus vs disagreement.

### Fair Uncertainty Quantification of Learned Representation (**WIP**)
The aim is to submit this work to a conference prior to the thesis submission. 

## Conclusion
The theme throughout this work is providing feedback to a designer.
The designer may be a data-collector, or an ML-practitioner.
This feedback is a form of interpretability that is missing from almost all fairness-inducing machine learning methods.
While the work in this thesis will surely not be the final word on this topic, important in-roads are made to allow
increasingly complex models to be deployed in consequential settings with confidence.

[^eccv2020]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A, 27% acceptance rate (2020).
[^cvpr2019]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A* (best), 25.2% acceptance rate (2019).
[^dami2021]: Impact Factor 3.670 (2020).
