# Research Area & Question

```{admonition} WIP
:class: Tip
This content is work in progress.
```

## Research Area

Machine Learning has been successful at a number of applications, in a number of contexts.
These applications include translation (in both image and natural-language domains), pattern recognition and decision making.
With contexts including Geology, Meteorology, Sports forecasting and Agriculture, to name a few.
Because of this success there is a desire to incorporate these systems in more and more situations, including situations directly applicable to people.
For example, Machine Learing systems have already been applied to police allocation, recidivism prediction, candidate screening and credit approval.
The promise is that instead of many human decision makers, each one biased with their own prejudices, heuristics and experience, we can have a uniform approach.
The hope is that by treating everybody the same, then unequal, biased behaviour can be removed.

Unfortunately, that's not always the case.

```{note}
There is a parallel to the Database saying "Garbage In, Garbage Out".
It's "Bias In, Bias Out".
```

There are a number of potential problems.
Examples of these problems include (but not limited to):

- _The tyranny of the majority:_ We optimise to be right for the many, at the expense of minority groups.
- _Sampling bias:_ We don't have representative data of our population.
- _Proxy labels:_ We don't (or can't) measure what we truly want to measure, so use a related quality as a proxy.
- _Biased data:_ The recorded human decision was just plain biased.

And unfortunately these aren't mutually exclusive.

To make matters more complicated, bias isn't just a property of data, it can also exist in model choice.
Whilst it's true to say that [a logistic regression model isn't biased](https://twitter.com/ylecun/status/1204008802086817792?s=20) the decisions around these models are made by people.
As such they may be affected by the decision maker's own biases.
There are also concerns of deployment setting, pre and post-processing and interpretation of the results.
However, these go beyond the scope of this work.

This work focuses on working with data, which is imperfect, but readily available, and asks two questions.

1. Given biased data, can we train a model to produce a fair representation of data from which an unencumbered downstream model would produce fair decisions?
2. If so, can we understand _what_ it was about the data that needed to change so that fair decisions can be made.

## Research Question
1. Can we learn a representation of data that makes downstream tasks **fair**?
   1. If so, can we understand what the representation **changed**?
