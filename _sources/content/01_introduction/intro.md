```{math}
\def\gA{{\mathcal{A}}}
\def\gB{{\mathcal{B}}}
\def\gC{{\mathcal{C}}}
\def\gD{{\mathcal{D}}}
\def\gE{{\mathcal{E}}}
\def\gF{{\mathcal{F}}}
\def\gG{{\mathcal{G}}}
\def\gH{{\mathcal{H}}}
\def\gI{{\mathcal{I}}}
\def\gJ{{\mathcal{J}}}
\def\gK{{\mathcal{K}}}
\def\gL{{\mathcal{L}}}
\def\gM{{\mathcal{M}}}
\def\gN{{\mathcal{N}}}
\def\gO{{\mathcal{O}}}
\def\gP{{\mathcal{P}}}
\def\gQ{{\mathcal{Q}}}
\def\gR{{\mathcal{R}}}
\def\gS{{\mathcal{S}}}
\def\gT{{\mathcal{T}}}
\def\gU{{\mathcal{U}}}
\def\gV{{\mathcal{V}}}
\def\gW{{\mathcal{W}}}
\def\gX{{\mathcal{X}}}
\def\gY{{\mathcal{Y}}}
\def\gZ{{\mathcal{Z}}}
```

# Chapter 1: Introduction 

The increasing capability of machine learning (ML) models to perform well at specific tasks has led to their use in more consequential applications.
This increased consequence has in turn led to greater scrutiny, with particular concern about what it means for an algorithmic decision, recommendation, or prediction to be 'fair'.
In response, the research community has begun investigating these questions which are grouped together under the term algorithmic fairness (AF).
This burgeoning field of algorithmic fairness has been the focus of a growing body of research, with a number of definitions being introduced to quantify and measure _un_-fair behaviour, which, as a research community we aim to minimise, or ideally, eradicate.
These definitions are often with respect to specific, legally protected characteristics that are observed alongside the features used for training an ML model, but cannot be used during inference.
Examples of these protected characteristics may include race, gender, age, or disability status, among others.

Although algorithmic fairness is a multi-faceted problem, this thesis investigates a specific instance of a general concern - that data can contain spurious correlations.
These are correlations that only appear in a subset of samples, but do not exist in the broader population.
Spurious correlations may be present in a subset of data used for training and validating an ML model, leading to a shortcut being exploited rather than a more complicated underlying true function being learnt.
This becomes particularly important when the spurious correlation is between the model target and a protected characteristic.
Examples of a model target for a classification task may be approval or not, for a hiring, loan or bail decision.
Simple rules such as 'invite male candidates to interview for a vacancy', or 'offer higher loans to white applicants' may perform well on the labelled training and validation data, but when deployed they may both perform poorly, and have the potential to cause significant harms to the population.
This specific type of spurious correlation, often referred to as _biased data_, is the source of concern in this thesis.
Biased data impacts performance and plays a large part in the trust afforded to ML-based systems.
The effect of this can have a significant impact, especially in the case of decisions that directly affect a person’s livelihood.

A promising approach to counteract biased data is by producing a fair representation (FR) as a pre-processing step.
In FR-learning, the aim is to produce a transformation of the data such that it still retains utility for a downstream task, but has been modified so that information about a protected characteristic of concern is either removed, or obfuscated to the point where a downstream model produces 'fairer' decisions by default.
The benefits of this approach are that the fairness-promoting aspect is isolated, allowing easier regulation, and allowing the process to be independent of other concerns.
However, this approach is not without drawbacks.
Completely removing some attributes while retaining utility is non-trivial; 
the burden of responsibility to check for unfair behaviour can be inadvertently moved away from a downstream system; 
and the data often becomes obscured when projected into an uninterpretable latent space, making intuitive assessment difficult.
Making progress in addressing some of these drawbacks may promote the adoption of fair representations and the benefits that they provide.


## Problem statement

This thesis investigates fair representations of data and whether they can be used to provide additional insight into a system.
Can we retain the benefits of fair representations of data - an isolated and measurable fairness-inducing intervention - while making progress in overcoming the shortcomings?
The result should be a transformation of the data that increases the fairness of a post-hoc ML model by default, while retaining the utility of the original input, and still remaining as interpretable as the input data.

This desiderata poses the research question that is tackled in this thesis:
'How can we make fair representations interpretable?'.
To approach this, I first develop a method that uses fair representations to interrogate the effect of a specific protected characteristic.
This will provide insight into the relationship between the feature of interest, the protected characteristic, and the remaining features.
I will demonstrate that this can be used to promote fairer outcomes without directly manipulating an existing decision system.

Secondly, I will demonstrate that fair representations can exist within the data domain itself.
This is not a trivial task. 
The resulting output of the transformation should reside in the original feature space and retain useful information about features other than the protected characteristic.
In addition, the transformation should also obfuscate that particular feature.

Lastly, I will improve on this first attempt at producing fair representations in the data domain, introducing models based on alternative approaches to achieve a more robust outcome with improvements to the qualitative results.


## Motivation and Aims
More data cataloguing human behaviour is being produced than ever before. 
The broad aim of many applications is to use this data to make sensible predictions about future events.
These can be to assist the user by preempting their needs and queries, or to make decisions about the effectiveness and cohesion of potential hires.
Ideally, to do this, we would aim to have the total information that was available to the user.
However, this is not realistic.
Instead, we typically have $n$ pairs of data, $(x,y)$ which form a dataset $D = \{(x_0, y_0), \dots, (x_n,y_n)\}$.
These pairs represent input features $x$ from the set of possible values $\gX$, and outcomes $y$ from the set of possible outcomes $\gY$.
If the data were total, then we would have all of the information necessary to emulate the true underlying mapping from $\gX$ to $\gY$. 
Instead, we are limited to obtaining, at most, data that can be recorded.
As such, the aim is not to reconstruct the ground-truth mapping, but instead produce an approximation.
Typically in ML we focus on finding an approximation function $f: x \rightarrow y$ from the set of possible functions in the hypothesis space $\mathcal{F}: \gX \rightarrow \gY$ that most accurately models this relationship (minimises the Empirical Risk).
However, recent works have questioned if this alone is the best criterion for success.
Instead, fairness-aware ML algorithms take into account additional information in the form of a protected characteristic, $s$, and seek to reduce the hypothesis space to functions that either don't make use of $s$ at all, or allow for a defined margin-of-error[^1].
An overview of related work that aims to achieve this is discussed in {ref}`ch:related_work`.

One application for ML models is emulating current decision processes.
For tasks such as loan approval, decisions have traditionally been made by a number of decision makers employed for the task, each with their own thresholds, preferences, and biases.
In such a setting, the promise of automated decision systems is clear.
An automated system can process millions of applications incredibly quickly, is available at all times, and crucially, will be consistent in its decision making process.
However, there are drawbacks.
Any errors or inconsistencies in the logic learned from observing past behaviour have a greater chance of being exposed, and worse, perpetuated.
With such a system deployed, it is no longer possible to pass off inconsistent decision making as human error.
The challenge to produce a fair system might be difficult, but there is significant opportunity for improvement from any unconstrained system. 
In such a system, making the outcome 'fairer' in any way can have a significant and practical impact, even if absolute fairness is not achieved.

One criticism that is often levelled at ML models, especially those deployed in human-centred scenarios, is that the decision making process is not clear.
In addition to our desire to produce fairer results, it is also important that stakeholders in the system feel confident in any fairness-enhancing interventions introduced.
On top of the aim of producing a fairer result, any amendments to the system should also increase the interpretability.
An improved fairness intervention solution would not only increase the fairness of the system, but would allow stakeholders to gain some knowledge of what changes are required for this to be met.

Lastly, a concern for generally adopting fair ML is the potential trade-off between model accuracy and how 'fair' the system is.
I explore more about different definitions of fairness in {ref}`ch:related_work`, however there is a simple case to demonstrate that there may not be a trade-off after all.
In cref\{1-intro:fig:imbalanced-s-skew\} we witness the case where the training dataset is imbalanced in relation to the deployment setting.
This can be for a number of reasons, such as using historical data, or only having access to a limited source of data.
In the deployment set, however, the data is balanced. 
If the features available to train a model are not sufficiently rich for a function to approximate the mapping of $\gX$ to $\gY$, but are sufficiently rich to map $\gX$ to $\gS$, then the the protected characteristic may become a _proxy label_ for the target.
In this case, the data could be categorised as biased - there exists a spurious correlation between $s$ and $y$ that is only present in the training set.
By providing an additional inductive bias that the outcome should _not_ be dependent on $s$, we may produce a function $f$ that is closer to the ground truth than the training data implies.

INSERT PIC HERE

The aims of this thesis are to provide an approach that combats specific spurious correlations in the form of biased data.
I produce this at the _pre-processing_ stage in the form of a data transformation.
While ideally a protected characteristic would be completely obfuscated, this is an unnecessary aim.
Instead, the task of deciphering the protected characteristic need only be more complex than learning to perform the task.

As an additional aim, the resulting transformation should give us some insight into the transformation process itself.
Of particular concern is additional problems being introduced by the transformation process. 
For example, if the protected characteristic is 'gender', then any changes made to hide this feature that in turn affect skin-tone are an indicator that the system designer may also need to consider 'race' as an additional source of potential bias.
Similarly, if the system returns a clearly degenerate solution, then it may save months of development time by highlighting this problem earlier.

## Claims and contributions
In this thesis, I produce three main contributions. 
Firstly, I demonstrate that fair representations can be used to produce fairer outcomes for already existing systems.
I achieve this by drawing a connection between the reconstruction of samples from fair representations and counterfactual examples. 
This work is catalogued in {ref}`ch:paper1`.

Secondly, I demonstrate that fair representations can exist within the data domain, making use of the inherent interpretability that this domain provides.
I make a first contribution to this in {ref}`ch:paper2` using a statistical dependence measure to promote a fairer representation under an additive decomposition assumption, allowing the data to be broken down to a 'fair' and 'unfair' component.

Lastly, I improve on this first attempt, assuming a more complex relationship between the 'fair' and 'unfair' components and introduce _null-sampling_ in {ref}`ch:paper3` to draw manipulated samples from a designated region of a learned latent space. 
This opens up alternative techniques to achieve fair representations in the data domain, making use of the properties of both a conditional Variational Auto Encoder (VAE) and an Invertible Neural Network (INN).


## Thesis Outline

This thesis is organised in the following way.
{ref}`ch:related-work` gives an overview of algorithmic fairness, and in particular, publications to date on fair representations of data.
Following this, {ref}`ch:content` describes each of the three main chapters of this thesis in greater detail, with an emphasis on how they relate to each other.
[Chapter 3](ch:paper1)-[Chapter 5](ch:paper3) contain three peer-reviewed and published works which have been reproduced with minimal changes except where explicitly indicated.
The work presented in {ref}`ch:paper1` is currently under review at **a journal**, however the conference proceedings on which the journal submission is based can be found in {ref}`app:paper1`.
{ref}`ch:paper2` contains an addendum with experimental results to help motivate {ref}`ch:paper3`.
Finally, in {ref}`ch:conclusion` I present the main conclusions and suggest possible future directions for the presented work.


[^1]: The maximum margin-of-error is often legally defined.
