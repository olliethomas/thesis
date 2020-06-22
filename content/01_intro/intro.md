# Introduction

![](https://img.shields.io/badge/status-DRAFT-critical)

From here onwards is a bit of a ramble, and very much WIP.
Please don't read too much into anything beyond this point.

## Fair Representations

This thesis looks at the relationship between data and outcomes.
Primarily in the case of outcomes in the form of decisions made using data.
Given some input data we use machine learning to create some function f that takes the data and produces some output.
This learnt function should be an approximation of the underlying relationship between the input and the output.
At best it is an approximation, however, some approximations are useful.
Our main concern in this thesis is that the model learnt is assumed to be useful, but in reality it causes harm.
This can be the case when the metrics used to assess the performance of the model appear to be good.
It may be that the accuracy (or some other metric) of the model is at a level determined to be good.
However, this is a metric to determine the performance for the population as a whole.
It can be the case that within our data is a subset of the population who are actively being harmed by this model.
For this group of people, the heuristic that the model has learned, the best approximation we have of the underlying function, is flawed.
It doesn’t work for them and it treats this group of people unfairly in comparison to the rest of the population.
This is a realisation of Simpson’s paradox.
Why does this happen in the first place?
Let's think about what this input data is and illustrate with an example.
The input data is some data which we have obtained which we believe may be useful to us in solving the task at hand.
In the case of the recidivism, the COMPAS system uses age, sex, location, number of previously committed crimes etc, in addition to some proprietary data which is unknown as a relevant input to a model.
There is a problem though, this data is limited. It is only obtained for those who are seeking parole and doesn’t represent the larger population.
Furthermore, we only record whether our system was correct for the number of individuals who we determined should be let out on parole.
Those who were determined to be too much of a risk to be let out, we don’t receive updates on.
Simply, we don’t know if we were right.
There are a number of problems here.
Maybe we shouldn’t use these systems in these scenarios?
Humans are flawed too, there was a psychologist mentioned in the weapons of math destruction book who is a good example of why these need to be brought in in the first place.
There’s also sampling bias occurring, this system is deployed in the US where the prison population is remarkably disproportionately represented by minority (in terms of the population) races.
We also expect residual unfairness in any learned system because of the masked, or unknown, outcomes.
Additionally, in this case, we have proxy labels (double check?).
One way to at least partly alleviate some of these problems is to massage the data, or adjust it in some way so that the data is less imbalanced.
There are a number of ways to do this, producing fair representations of the data.
These representations should be invariant to some sensitive attribute.
In fact, it shouldn’t be possible to reconstruct sensitive information from the representation, but it should still be expressive.
This creates tension.
The most “invariant” representation would be to have all data points mapped to the same output, however this will not be useful for a downstream task.
The least “invariant” representation would be to use an identity function for the data (maybe concat the sensitive attribute?).
This would be at least as informative as the original data for a downstream task, but it wouldn’t obfuscate any information.
Furthermore, in regard to fair representation, the output is in a non-interpretable space.
One of the worries of machine learning is that we are using a “black-box” to make decisions without explanation.
In the case of fair latent representations we have introduced another “black-box”.
