# Introduction

This thesis is ultimately about exerting additional control over a machine learning model.
Instead of allowing the simplest possible function to be learnt, we want to encourage

## Overview

### The Goals of Machine Learning
Given inputs $x \in \mathcal{X}$ and a target $y \in \mathcal{Y}$, the goal is to learn a function $f: x \rightarrow y$ that emulates some underlying relationship.

### Attributes and Features

Stratified sampling?

### Sensitive Attributes
Not all attributes are equally valid.

###  Human Biases
People can make inconsistent decisions. 
Sometimes these are not based on an objective measure.
Large number of decision-makers, then of course you will have a greater number of inconsistent decisions.
The promise of an automated decision-maker is that the decisions should be consistent.
New instances of the decision-making system can be spun-up as required to handle large numbers of decisions.
The system, once running, is cheaper than hiring many people.

### Accuracy as an objective
Goodhart's Law: When a measure becomes a target, it ceases to be a good measure.
If we optimise just for accuracy, then we will get a model that is overly narrow. 
It is good at that specific-thing. 
It will not necessarily capture the values of our organisation during the decision-making process.

#### Simpson's Paradox
A trend can exist in subgroups of the data, but the trend can be reversed when these groups are combined.
In relation to the previous section, when acuracy is improved overall, it can be that the accuracy for certain seubgroups is actually reduced.


#### Feedback Loops
We should consider systems over time. 
Humans aren't static and the result of a decision system can have an effect on an individual.
Consider a credit decision system. 
If an applicant is rejected for a loan, then this will affect their credit rating, making it harder in the future to pass a credit decision-making step.

#### Pareto Optimality
When there are competing objectives, it's often that the "correct" answer is not obvious.
In fact, there can be multiple "correct" answers.
The Pareto Frontier is collection of outputs, where each is the optimal output for that speicific balance between $$n$$ outcomes.


## Fair Machine Learning

### Pre-process

### In-process

### Post-process

### Metrics

### Datasets

## Definitions
Broadly, definitions of fairness can bve split into two sections: Group and Individual notions of fairness.


### Group Notions of Fairness

#### Independence

#### Separation

#### Sufficiency

### Individual Notions of Fairness

#### Individual Fairness

#### Counterfactual Fairness

### Competing Definitions

## Challenges
Need to record $S$ so that we can be fair with regard to $S$.

Distribution shift. 
Particularly over time in systems with long feedback loops.

Learning from biased data.
Walter Quijano, a pschologist testifying in the trial of Duane Edward Buck, asserted that "African Americans pose a greater risk of 'future dangerousness.'"
Without focussing too much on that case, it should be relatively clear that a system trained on data such as this may learn to repeat some of the behaviours.
[link](https://www.huffingtonpost.co.uk/entry/supreme-court-death-penalty-case-duane-buck_n_1080112)

Optimising without taking fairness metrics into account.

## Background

## Adversarial Methods

## Distribution Matching

## Counterfactual Fairness
