# Research Area & Question

## Research Area

Machine Learning is a tool that is growing in popularity.
There have been a number of high profile successes, and new applications are being regularly identified.
These applications include translation (in both image and natural-language domains), pattern recognition and decision-making.
Contexts for these applications include Geology, Meteorology, Sports forecasting and Agriculture, to name a few.

Because of this success there is a desire to incorporate these systems in more and more situations, including those directly applicable to people.
For example, Machine Learning systems have already been applied to police allocation, recidivism prediction, candidate screening and credit approval.

On top of the benefits of automated decision making (speed, scale, etc) there is an additional promise to automated decisions.
The promise is that instead of many human decision makers, each one biased with their own prejudices, heuristics and experience, we can have a uniform approach.
The hope is that by treating everybody the same, then unequal, biased behaviour can be removed.

Unfortunately, that's not always the case.

Recent headlines include:
- **Wrongfully Accused by an Algorithm**: In what may be the first known case of its kind, a faulty facial recognition match 
led to a Michigan man’s arrest for a crime he did not commit. -- NYTimes {cite}`Hill20`
- **Amazon ditched AI recruiting tool that favored men for technical jobs**. -- The Guardian {cite}`Reu18`
- **Police officers raise concerns about 'biased' AI data**. -- BBC News {cite}`BBC19`

But how does this happen?
A prediction model has to be designed and there are a number of legal and moral obstacles to prevent a group/individual from purposefully designing a biased system.
However, even for the best intentioned there are a number of potential problems.
Examples of these problems include (but are not limited to):

- _The tyranny of the majority:_ We optimise to be right for the many, at the expense of minority groups.
- _Sampling bias:_ We don't have representative data of our population.
- _Proxy labels:_ We don't (or can't) measure what we truly want to measure, so use a related quality as a proxy.
- _Biased data:_ The recorded human decision was just plain biased.

And unfortunately these aren't mutually exclusive.

An unconstrained machine learning model is susceptible to all of these problems.
To face this challenge, the machine learning community has focused on creating a class of learning models that are constrained 
to exhibit less bias than an unconstrained model.
Typically, these are referred to as “Fair Machine Learning Models”.

### Fair Machine Learning

Fair Machine Learning, in general, aims to resolve "unfairness" (or the effect of bias) by affecting a model in one of 
three places in the general learning pipeline.
1. Pre-model training.
    - Using a model to reduce bias in the underlying data, so that an unconstrained downstream classifier will exhibit fairer outcomes.  
2. During model training.
    - Adding constraints to a model so that breaches in these constraints are heavily penalised during training.
3. Post-model training.
    - Adjusting the output of an unconstrained model so that the adjusted outputs don't breach a given constraint.

| Approach      | Multi-task support    | Task model agnostic   | Pareto-optimal | Constraints guaranteed to be enforced | 
| :---          | :---:                 | :---:                 | :---:          | :---:                                 |
| Pre-process   | X                     |  X                    |                |                                       |
| During        |                       |                       | X              |                                       |
| Post-process  |                       |  X                    |                | X                                     |

```{figure} ../assets/fairness_taxonomies.png
---
width: 600px
name: Fairness Taxonomy
---
Taxonomy of fairness methods {cite}`Keh21`
```

In my work I focus on removing bias in the 'pre-processing' stage. 
This is an exciting and active research area with spotlight tutorials at top conferences such as NeurIPS[^footnote] {cite}`CisKoy19`.
Part of the reason for the excitement in this area is that the underlying data itself is a major source of bias.
After all, this is what a model is using to determine "correct" behaviour.
If we are able to understand and counteract bias that exists in the underlying data, then we can use an unconstrained 
classifier (which may already have been heavily invested in) for a task.
Crucially, we may be potentially able to use the same data for performing _multiple fair tasks_.

Ultimately though, even a model counteracting bias will not be fully trusted without being able to interpret the actions 
that it has taken to counteract bias.
Simply off-loading a problem from one black-box to another only masks the issue.
My work specifically deals with _this_ problem.

% This isn't to discount bias entering in other manners.
% Bias isn't just a property of data, it can also exist in model choice.
% Whilst it's true to say that [a logistic regression model isn't biased](https://twitter.com/ylecun/status/1204008802086817792?s=20) 
% the decisions around these models are made by people.
% As such they may be affected by the decision maker's own biases.
% There are also concerns of deployment setting, data collection and interpretation of the results.
% However, these go beyond the scope of this work.
% 
% ```{note}
% There is a parallel to the Database saying "Garbage In, Garbage Out".
% In Machine Learning we have "Bias In, Bias Out".
% ```

This work focuses on working with data, which is imperfect, but readily available, and asks:

Given some data, can we use machine learning to provide feedback about changes needed in the data, or an existing system, to counteract specific types of bias?

[^footnote]: http://sanmi.cs.illinois.edu/documents/Representation_Learning_Fairness_NeurIPS19_Tutorial.pdf

## Research Question

1. Can we learn a representation of data that makes downstream tasks **fair**?
   - If so, can we provide feedback to the model designer.
   
I will answer this question by: 
- demonstrating that fair representations of data can be built in the original data domain without loss of performance with regard to both utility and fairness criterion.
  - this will provide feedback as to how the model _changes_ the data so that it becomes fair.
- demonstrating that fair representations can be used to estimate counterfactual reconstructions of the input.
  - this will provide feedback as to which _individuals_ are at risk of an unfair decision, by querying if they would receive
    a fair decision in a counterfactual world.
- demonstrating that fair representations can be used to quantify uncertainty in a decision.
  - this will provide feedback as to how confident a model is that the decision is fair.
 

```{bibliography}
:filter: docname in docnames
```
