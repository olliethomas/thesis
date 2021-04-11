# Introduction

This thesis is ultimately about exerting additional control over a machine learning model.
Instead of allowing the simplest possible function to be learnt, we want to encourage learning truth.
We want our models to find complex, but true answers.
Not simple convenient heuristiics.

![](https://i.imgur.com/F2Y91nd.jpg)

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

### Interpretability

Interpretability is the degree to which a human can understand the cause
of a decision. Some models are inherently interpretable, such as
Decision Trees, or to a lesser extent linear models. There are also
model agnostic interpretability techniques such as local surrogate
models such as "Local Interpretable Model-agnostic Explanations (LIME)"
{cite}`RibSinGue16`, or game-theory approaches to explanation
such as the Shapley Value {cite}`Roth88`.

From "Explanation in Artificial Intelligence : Insights from the Social
Sciences" {cite}`Miller19`, Interpretability is the degree
to which a human can understand the cause of a decision. The book
"Interpretable Machine Learning" {cite}`Mol18` frames that statement in a
slightly different way, describing interpretability is "the degree to
which a human can consistently predict the model's result". In this
review we have already distinguished between transparency and
interpretability, but {cite}`Mol18` distinguishes further between
interpretability and explanation. Miller {cite}`Miller19`
argues that even if we are capable of interpreting the results of a
model, unless we receive an explanation of how that model came to make a
decision, then we will be unable to reliably reproduce the results. not
only that, but as humans, any old explanation will not do, we require a
_good explanation_. According to Miller {cite}`Miller19`,
good explanations are:-

- **contrastive**. We tend to think in a counterfactual way i.e. would
  I have been approved for a loan if I earned more money. Explanations
  should reflect this.

- **selected**. The world is complex and we don't like to receive too
  much information. As such, we should only give 1 to 3 explanations
  that cover the majority of cases.

- **social**. They should be tailored to your audience.

- **focused on the abnormal**. If a data-point contains an anomaly
  that impacts the result, use that in the explanation. i.e. house
  price being predicted unusually highly as the property contains 5
  balconies.

- **truthful**.

- **coherent with prior beliefs of the explainee**

- **general and probable**


## Fair Machine Learning

Fair Machine Learning, in general, aims to resolve "unfairness" (or the effect of bias) by affecting a model in one of 
three places in the general learning pipeline.
1. Pre-model training.
    - Using a model to reduce bias in the underlying data, so that an unconstrained downstream classifier will exhibit fairer outcomes.  
2. During model training.
    - Adding constraints to a model so that breaches in these constraints are heavily penalised during training.
3. Post-model training.
    - Adjusting the output of an unconstrained model so that the adjusted outputs don't breach a given constraint.

````{margin}
```{note}
This comparison is of course a little over-simplistic.
``` 
````
| Approach      | Multi-task support    | Task model agnostic   | Pareto-optimal | Constraints guaranteed to be enforced | 
| :---          | :---:                 | :---:                 | :---:          | :---:                                 |
| Pre-process   | X                     |  X                    |                |                                       |
| During        |                       |                       | X              |                                       |
| Post-process  |                       |  X                    |                | X                                     |

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

### Approaches

As with all areas, the lines between the various points that fairness
constraints have been injected isn't an easy thing to split. For
example, at what point does pre-processing the data become learning a
new representation for the data? As such, some of these methods blur
boundaries. However, in general, we can think of fairness interventions
occurring before, during, or after the training of the model as
suggested in {cite}`BarHarNar19`.

In this section we'll look onto the broad categories *pre-processing* -
which includes feature selection and feature adjustment, *during
training* - which covers minimising fairness constraints directly,
adversarially removing sensitive attributes and learning fair
representations and *post-processing* - which hasn't been well utilised,
but is still a valid point to insert fairness constraints.

#### Pre-process
This can form in two approaches. On one hand, the features can be
pre-processed, or the labels can be pre-processed. There is a general
trend to transforming the features to a new space, called learning a
fair representation of the data. This is covered under the 'During
Training' section. This section is reserved for those papers which
explicitly change the input features, such as "Certifying and Removing
Disparate Impact" {cite}`FelFriMoeSchVen15`. This paper compares
the probability distributions of features across protected groups and
seeks to rectify this. Enforcing

$$
P(\bar{x}|s=0) = P(\bar{x}|s=1)
$$

where $\bar{x}$ is a modified version of the original feature $x$.

In this approach, each feature is ranked within sensitive sub-groups,
then shifted so as to retain the same rank whilst having an identical
distribution for each sub-group such that $P(s|\bar{x}) = 0.5$.

The concept of changing the input space is an intriguing one. On one
hand, a person's features are not discriminatory in themselves, after
all, discriminatory features are _usually_ qualitative. That said, some
features are due to the disparate impact of previous unfair decisions.
An example of this may be disproportionate wealth distribution in the US
manifesting itself as an unusually small number of white people
receiving public schooling
{cite}`USCBQ,USNCES`. Modifying this feature
to correct for previous biases may have merit. It may also be useful to
consider that amending the input space allows us to interpret what has
changed about an individual to make them appear less worthy of being
biased against, and to feed this back to policy makers. This is under a
reasonable assumption that the input space records some features that we
are able to interpret. For example, if we knew that an applicant were
more likely to pass the CV screening stage of a job application in a
decision agnostic to their gender by increasing the level of relevant
work experience, it may be that there should be a policy change to
encourage people into apprenticeships or internships. The other approach
is changing the labels, as discussed in section
[2](#background){reference-type="ref" reference="background"} about the
Kamiran and Calders paper {cite}`KamCal09`.

#### In-process

One of the more direct ways to enforce fairness is to follow Quadrianto
et al {cite}`QuaSha17` who noticed that enforcing
fairness constraints is an application of Distribution Matching. They
use a modified SVM from the Learning Using Privileged Information (LUPI)
paradigm to ensure the sensitive feature is not used during test time,
but is available during training. They then pose a question about how
much fairness to apply. There is a fairness-utility trade-off and the
authors suggest a human should be responsible for selecting how much
fairness (within a legal limit) to apply. This concept of bringing
accountability into automated decision making is an important though
overlooked addition.



#### Post-process

This area isn't as well explored, but {cite}`HarPriSre16` use it in their
paper.

#### Metrics

### Datasets

### Definitions
Broadly, definitions of fairness can bve split into two sections: Group and Individual notions of fairness.


#### Group Notions of Fairness

Fairness constraints on the other hand have three forms. In the
currently unfinished book on Fairness in Machine Learning
{cite}`BarHarNar19` the definitions of fairness are described as
belonging to one of three groups, _Independence_, _Separation_ or
_Sufficiency_. This pattern is adopted in this section.

##### Independence

The most intuitive notion of fairness is _Independence_. This is the
notion that given a prediction ($\hat{Y}$) and a protected sensitive
attribute ($S$), then the prediction should be independent of the
sensitive attribute 

$$P(\hat{Y}) \perp S$$ 

In fact, one of the first
papers in this area by Kamiran and Calders {cite}`KamCal09` use this as
their discrimination measure. Although written in a different form, they
later use the notation latterly adopted in fairness literature during
their journal article {cite}`KamCal12` expanding on their previous work
{cite}`KamCal09,kamiran2011`

$$disc = P(\hat{y}|s=0) - P(\hat{y}|s=1)$$ 

Statistical Parity (or
Demographic Parity as it often known) appeals to an intuitive sense of
group fairness, namely that the outcome of the model should be
independent of some sensitive attribute(s). For example, the probability
of you being accepted at university should be the same if you are male
as if you are female (and vice-versa).

There are situations however, where this doesn't work as intended. In
these cases, instead of promoting the perceived harmed group based on
the quality of the individual, as long as the probability of acceptance
is the same, the criteria is met.

To illustrate this, let's consider an example that will be used across
all our definitions. Imagine that we are in charge of admissions at a
university and we are particularly concerned with complying to a
fairness criteria with regard to male and female subgroups. At this
university we can only accept 50% of all applicants. To determine if you
are likely to succeed there is an entrance criteria, which is highly
predictive of success. In fact, 80% of people who meet the entry
criteria successfully graduate. However, many students apply despite not
meeting the criteria. Universally, only 10% of students successfully
graduate if they do not meet the entrance criteria. Both male and female
subgroups apply to our university in equal numbers, though only 40% of
applying males meet the entrance criteria, whilst 60% of applying
females meet the entrance criteria. As already stated, we can only
accept 50% of applicants to be students. Under Demographic Parity, we
would require that 50% of both males and females be accepted, regardless
of likely academic performance. So even though only 40% of male
applicants meet the qualifying academic criteria, an additional 10% of
the population would have to be accepted at random to be 'fair', whilst
10% of qualified females would be rejected.

To counter this, yet keeping within the frame of _independence_,
relaxations of this criterion have also been suggested to include parity
up to some threshold, $\epsilon$. {cite}`ZafValGomKri19`

$$P(\hat{Y}|s=a) \geq P(\hat{Y}|s=b) - \epsilon$$

or via a ratio

$$\frac{P(\hat{Y}|s=a)}{P(\hat{Y}|s=b)} \geq 1 - \epsilon$$

When set to $\epsilon = 0.2$, this is seen as comparable to the _80%
rule_ mentioned in disparate impact law {cite}`FelFriMoeSchVen15`.
This rule suggests that as long as the selection rate of the 'harmed'
group is within 80% of the 'privileged' group, then it is fair enough.
Though critics of this point out that 80% was chosen arbitrarily.

##### Separation

A more complex definition of fairness is _separation_, which is
independence given the actual outcome ($Y$) $$\hat{Y} \perp S | Y$$

This has been formalised by the metric Equalised Odds {cite}`HarPriSre16` which
considers all values of $Y$, and the looser constraint Equality of
Opportunity {cite}`HarPriSre16`, which only constrains independence given the
outcome is positive .

Equalised Odds 
```{math}
\begin{align*}
    P(\hat{Y}|s=0, y=0) &= P(\hat{Y}|s=1, y=0) \\
    \& \\
    P(\hat{Y}|s=0, y=1) &= P(\hat{Y}|s=1, y=1)    
\end{align*}
```

Equality of Opportunity $$P(\hat{Y}|s=0, y=1) = P(\hat{Y}|s=1, y=1)$$

Essentially, Equalised Odds ensure matching true positive rate (TPR) and
false positive rate (FPR) across the sensitive groups, whereas Equality
of Opportunity only ensures that the TPR of both (all) sensitive groups
are equal. The benefit of this is that it is a truer representation of
fairness.

In our university admission example, Equality of Opportunity is
equivalent to accepting members of both female and male subgroups at
different rates, as long as the true positive rate of both groups is
equal. So we would be looking to accept $44.5\%$ of males and $55.5\%$
of females, which would give both groups a TPR of 85.4%.

If we were enforcing Equalised Odds, we would have to make sure that we
were not only matching the true positive rate, but also the false
positive rate. In our example, the selection rate would be $46.4\%$ for
males and $53.6\%$ for females.

But could an algorithm satisfy both independence and separation?
Unfortunately not, it is shown in {cite}`HarPriSre16` that independence based
notions of fairness are incompatible with separation based notions of
fairness.

##### Sufficiency

There is a less well utilised general area of fairness criteria called
_sufficiency_. This is the concept that the true outcome, given the
predicted score is independent of $s$.

$$Y \perp S | \hat{Y}$$

In our example, we would leave the selection rates alone, giving a
selection rate of 40% for the male subgroup and 60% for the female
subgroup as then we treat applicants equally based our anticipation of
their success.

Similarly to the conflict between independence and separation, there is
a conflict between independence and sufficiency, and between separation
and sufficiency. These conflicts can be described succinctly, so we
demonstrate them below.

To show that sufficiency and independence are mutually exclusive, let's
assume that there exists data where $Y$ is dependent on $S$.
Independence is $\hat{Y} \perp S$, sufficiency is $Y \perp S | \hat{Y}$.
By simply claiming both of these statements to be true, we then have

$$
S \perp \hat{Y} \textbf{ and } S \perp Y | \hat{Y} 
\Rightarrow S \perp Y
$$

This shows that independence and sufficiency can only hold when we
contradict our assumption.

Similarly, for separation and sufficiency if we assume that again we
have data where $Y \not \perp S$ we can show that for both to hold we
would have

$$
S \perp \hat{Y} | Y \textbf{ and } S \perp Y | \hat{Y}
\Rightarrow S \perp Y
$$

which again is contradictory to our assumption, demonstrating that we
can't have both sufficiency and separation based notions of fairness
unless the data is inherently unbiased. This notion of trade-offs and
balancing tension is a common one throughout fair machine learning, and
certainly one that we will come back to.

#### Competing Definitions
The above is a useful framework for viewing fairness constraints and
helps us to categorise various definitions of fairness, such as those in
table 1, but that shouldn't diminish work that seeks to make novel
strides within each of the areas. For example, the work in "An
Intersectional Definition of Fairness"
{cite}`FouIslKeyPan20` expands the independence notion of
fairness. Their inspiration comes from third-wave feminism and
intersectional privacy, and is trying to expand beyond binary sensitive
groups and measuring an unfairness value at each intersect. For example,
consider we have a dataset with three sensitive attributes, sex, race
and religion. Most approaches to date consider these to be one feature
_sex_race_religion_. This paper measures the difference with respect
to demographic parity between each combination of sensitive attributes
so that sex, race and religion are all viewed as separate, measurable
points of potential discrimination, being concerned with whether
discrimination occurs in any, some, or if only with all attributes
present.

| Fairness Goal                         | Definition                                                                                              | Example of   | 
| ------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------ |
| Demographic Parity                    | $P(\hat{Y} \| S = 0) = P(\hat{Y} \| S = 1)$                                                             | Independence |
| Equal Opportunity                     | $P(\hat{Y} = 1 \| Y = 1, S = 0) = P(\hat{Y} = 1 \| Y = 1, S = 1)$                                       | Separation   |
| Equalised Odds                        | $P(\hat{Y} = 1 \| Y = y, S = 0) = P(\hat{Y} = 1 \| Y = y, S = 1) \forall y \in Y$                       | Separation   |
| Equal Accuracy                        | $P(\hat{Y} = Y \| S = 0) = P(\hat{Y} = Y \| S = 1)$                                                     | Independence |
| Accurate Coverage                     | $P(\hat{Y} = 1 \| S = s) = P(Y = 1 \| S = s) \forall s \in S$                                           | Sufficiency  |
| Not Worse Off / No Lost Accuracy      | $P(\hat{Y}_{\text{new}} = Y \| S = 1) \geq P(\hat{Y}_{\text{current}} = Y \| S = 1)$ | Separation   |
| No Lost Benefits / No Lost Coverage   | $P(\hat{Y}_{\text{new}} = 1 \| S = 1) \geq P(\hat{Y}_{\text{current}} = 1 \| S = 1)$ | Independence |
Differing Fairness Criterion and their categorisation

Which fairness definition (or family of definitions) should one be
using? This is a more complex question to answer. Some, such as
{cite}`YeoTsc21,HeiLoiGumKra19` make the assumption the the choice of which
to apply should come from the designer's moral perspective, arguing that
this is a task outside of the expertise of computer scientists and
instead should be debated by philosophers.

However, not everyone is agreed. Recently there has started to be work
investigating the delayed impact that fair interventions in machine
learning have on society {cite}`LiuDeaRolSimetal18`. This work looks to determine the
impact that different notions of fairness have on the groups involved,
recognising that there is more than the initial 'accepted for a loan' or
'rejected for a loan' dichotomy, but that this has an impact in terms of
credit score for the individual applying. This is a bold approach that
tries to measure the effect that automated decisions may have 1
generation into the future. Although not explicitly stated, it leaves
the suggestion that maybe we should be using statistical models to
underpin moral assumptions.

A recent paper "Evaluating Fairness Metrics in the Presence of Dataset
Bias" {cite}`HinCooMamDee18` looked into the problem of
determining which fairness criteria to apply. They look at a dataset[^5]
which they use to create 4 datasets, with combinations of Sample Bias,
No Sample Bias, Label Bias and No Label Bias. They consider a binary
race attribute (Black, White) where White race is $s=0$ and Black race
is $s=1$. Label Bias is where there are different label thresholds based
on race, in this case 

$$
\hat{Y} = 
\begin{cases}
1 \text{ if score} \geq 0.3, \text{else } 0 & \text{if race = white} \\
1 \text{ if score} \geq 0.7, \text{else } 0 & \text{if race = black}
\end{cases}
$$

Sample Bias is where one group (in this case White race) has people
selected at a higher rate, in this case

$$
\tilde{P}(x \in \mathcal{X}) = 
\begin{cases}
0.8 & \text{if race = white and score} \geq 0.5 \\
0.2 & \text{if race = white and score} < 0.5 \\
1   & \text{if race = black}
\end{cases}
$$

This paper demonstrates that no single fairness metric is able to pick
up all discrimination and that all fairness metrics require "a healthy
dose of human judgement".

#### Individual Notions of Fairness

All definitions of fairness that we've looked at so far consider
proportions of groups, called *group fairness*. There is another
approach, called *individual fairness*. This is the idea that regardless
of any groups as a whole, similar individuals should be treated
similarly. This idea was proposed by Luong et al.
{cite}`LuoRugTur11` in the context of k-NN and later refined by
Dwork et al. {cite}`DwoHarPitReiZem12`.

Luong et al. use a Manhattan distance of z-scores for interval-scaled
attributes, and percentage of mismatching values for nominal attributes
to determine the distance between data-points. They determine that
discrimination has occurred if in its k-nearest neighbours those within
the same protected category have been treated differently to the
neighbours of a different category. They propose that on finding points
where they are confident that some discrimination has occurred, then the
class-label for that point should be amended. This data should then be
used to train subsequent models.

A seminal paper in the field, Dwork et al. {cite}`DwoHarPitReiZem12`
continued with the concept of a distance measure. They propose again
that individual fairness should be the goal. They suggest that given
some *task-specific* similarity metric, $\delta$, then
$\forall x, x' \in \mathcal{X}: \quad |f(x) - f(x')| \leq \delta (x, x')$
where $f(x)$ produces a continuous score as opposed to a discrete label.
The authors acknowledge that obtaining $\delta$ is a tricky problem,
described as "one of the most challenging aspects of this framework". It
may require input from social and legal scholars or domain experts to
help formulate this metric. This paper is particularly notable because
each of the authors have gone on to write about fairness extensively,
each gaining a reputation as an expert within their own right and covers
themes that have been spun into more fleshed out ideas. Examples of this
are Demographic Parity, the predominant fairness measure at the time,
being weak in some situations which has been explored in
{cite}`HarPriSre16,LiuDeaRolSimetal18`, and noting that it there is a differentiation
between a data-provider (who obtains the data) and a vendor (who uses
the data to make decisions). It's discussed that the vendor may not care
about fairness and this is explored further by two of the authors in
{cite}`ZemWuSwePitDwo13,MadCrePitZem18`. This is
discussed in section [4.3](#impl_repr){reference-type="ref"
reference="impl_repr"}.

##### Individual Fairness

##### Counterfactual Fairness

With the acknowledgement that fairness is difficult to solve
arithmetically, methods to incorporate subject matter expertise are
being explored. Causal models are appealing in this regard because they
allow for an explicit causal relationship to be accounted for rather
than relying on correlation.

DeDeo {cite}`DeDeo14` argues that without understanding the
causal relationship between attributes, then it becomes particularly
difficult to differentiate between innocent relationships, and those
which at first glance may appear innocent, but when you understand the
socio-economic background of those attributes, they might infer a less
innocent relationship.

Stemming from the work of Pearl {cite}`Pearl09`, causal models
are an attempt to model cause and effect. An example would be
atmospheric pressure and the position of the needle on a barometer
reading. We know that the two are linked and our data about this will
demonstrate a high correlation between observations, but correlation
does not imply causation. Whist we know that changing the pressure will
effect the barometer, moving the needle on the barometer will not effect
the pressure in the room. The benefit of viewing the world in this way
is that we can transparently interpret why decisions have been made.
Obviously the relationships between features is complex, but we can
utilise experts from the domain we are trying to apply our model to.
This is a nice feature given that fairness itself is domain specific.
Whilst this may seem simple on the surface, it is highly complicated to
correctly model the world. For example, not all features are captured.
There may be an unobserved feature that confounds two features, so
whilst they may look as though they are connected in some way, they are
actually both reflective of the unseen confounder. An example of this is
height and level of education. On the surface we could draw a
correlation that the taller (on average) a population is, the higher the
level of education. This can be observed by visiting any primary or
secondary school, but we're missing a confounder, age. What's more,
there can be multiple confounders that affect different sets of
features. Whilst not insurmountable, this is nonetheless a very labour
intensive approach. In many ways, if this approach is fully realised, it
is the gold standard for ethical models.

A recent paper, "Path-Specific Counterfactual Fairness"
{cite}`Chi19` poses some thought provoking
questions. They note that not all affects of a sensitive attribute on
the outcome are potentially discriminatory. They give the example of the
Berkley admissions data that was suggested to be discriminatory to
women. They note that women were applying with greater proportions to
classes with low acceptance rates, thus the influence of gender on the
class applied for is not discriminatory and should be taken into account
to learn a highly predictive model. This is similar to the idea first
mentioned in Pedreschi et al. {cite}`PedRugTur08`
that there is a difference between sensitive attributes and potentially
discriminatory attributes. In {cite}`Chi19` they
use the power of a causal model to isolate this to effects along
specific pathways noting "approaches based on statistical relations
among observations are in danger of not discerning correlation from
causation, and are unable to distinguish the different ways in which the
sensitive attribute might influence the decision"
{cite}`Chi19`. This paper views unfairness as the
presence of an unfair causal effect of $S$ on $\hat{Y}$. This idea is
not new. In fact it is specifically mentioned in "Counterfactual
Fairness" {cite}`KusLofRusSil17` that "a decision is unfair
toward an individual if it coincides with the one that would have been
taken in a counterfactual world in which the sensitive attribute were
different". This assumes that the entire effect of $S$ on $\hat{Y}$ is
problematic. The path-specific approach uses the same definition, but
modifies the ending to be "\... counterfactual world in which the
sensitive attribute _along the unfair pathways_ were different". They
achieve this by measuring the effect of $s$ along unfair pathways and
disregarding it. In the simple case below

```{figure} ./assets/path-specific.png
---
height: 200px
name: path-specific
---
```

where the direct effect of $s$ on $y$ is fair, but the effect of $s$ via
$m$ is unfair, then we can think of each variable being created of it's
own characteristic $\theta^{\text{variable}}$, plus the effect of its
parents $\theta^{\text{variable}}_{\text{induced by}}$, plus noise
$\epsilon_{\text{variable}}$, so

```{math}
\begin{align*}
S &= \theta^s + \epsilon_s \\
M &= \theta^m + \theta^m_s + \epsilon_m \\
Y &= \theta^y + \theta^y_s + \theta^y_m + \epsilon_y
\end{align*}
```

our goal would be to remove the effect of $s$ along unfair pathways,
giving

```{math}
\begin{align*}
S &= \theta^s + \epsilon_s \\
M_{\text{fair}} &= \theta^m + \epsilon_m \\
Y_{\text{fair}} &= \theta^y + \theta^y_s + \theta^y_{m_\text{fair}} + \epsilon_y
\end{align*}
```

In this case, (and in the case of "Counterfactual Fairness"
{cite}`KusLofRusSil17`). The goal is to achieve fairness in
a counterfactual world as described above. This is a form of individual
fairness, and is often linked to the other seminal work in this area of
Dwork et al. {cite}`DwoHarPitReiZem12`. Whilst one uses a causal
model to determine the effect of group membership, the other uses a
distance measure. Clearly there are strengths and weaknesses to both
approaches.


### Challenges
Need to record $S$ so that we can be fair with regard to $S$.

Distribution shift. 
Particularly over time in systems with long feedback loops.

Learning from biased data.
Walter Quijano, a pschologist testifying in the trial of Duane Edward Buck, asserted that "African Americans pose a greater risk of 'future dangerousness.'"
Without focussing too much on that case, it should be relatively clear that a system trained on data such as this may learn to repeat some of the behaviours.
[link](https://www.huffingtonpost.co.uk/entry/supreme-court-death-penalty-case-duane-buck_n_1080112)

Optimising without taking fairness metrics into account.

## Background

### Adversarial Methods

### Distribution Matching

### Counterfactual Fairness


## Research Question

1. Can we learn a representation of data that makes downstream tasks **fair**?
   1. If so, can we understand what the representation **changed**?
   
I will answer this question by: 
- demonstrating that fair representations of data can be built in the original data domain without loss of performance with regard to both utility and fairness criterion.
- demonstrating qualitatively that representations in the data domain provide feedback as to what is required to make data “fair”.

[^footnote]: http://sanmi.cs.illinois.edu/documents/Representation_Learning_Fairness_NeurIPS19_Tutorial.pdf
[^1]: Certainly from the ethical perspective, although interpretability of machine learning models is a research area in it's own right.
[^2]: And alternatively, we may be able to interpret some stages of the decision making process but the process to obtain these may be opaque to us.
[^3]: For more on this paper, see {ref}`causal`.
[^4]: Throughout this section we consider a binary sensitive attribute, though all definitions can be extended to accommodate a sensitive attribute with $n$ categories.
[^5]: Which due to contractual reasons they are unable to release.
[^6]: In this case, with regard to demographic parity.
[^7]: The CelebA, Soccer Player and Quickdraw datasets were used.
[^8]: Given the arbitrariness of many 'fairness' rules in the U.S. that have led to claims of disparate treatment (i.e. Ricci vs DeStefano or Texas House Bill 588) this objective may have a large impact.
[^9]: And consequently the expectation to be free of discrimination based on that religion.
---
```{bibliography}
:filter: docname in docnames
```
