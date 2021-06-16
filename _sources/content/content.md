# Content

This site serves as an expanded thesis/collection of works, and is structured in the following way.

First, the work-in-progress thesis, followed by annual review documents for reference, and lastly there is an appendix of works completed.

## Thesis

The structure of the thesis doesn't fall into a thesis by publication, but rather, a thesis supported by publication.
By that, I mean that I intend to have full thesis chapters that expand on published works.
The "story" that I am trying to tell is about identifying the parts of a decision making process that are influenced, 
rightly or not, by a sensitive attribute.

## [Chapter 1](02_data_domain_fairness/intro.md)
To identify these, we begin by trying to separate the "sensitive" part of a datapoint that have an effect on the decision
making process, from the parts of a datapoint that are not effected by a "sensitive" attribute.
This is achieved in [](09_appendix/publications/dfritdd.md), with a simple assumption of additive decomposition.
This is then extended in [](09_appendix/publications/nosinn.md) to allow for an arbitrarily complex reconstruction.

In both of these works we increase the _transparency_ of what the model determines to be related to a sensitive attribute by visualising the embedding within the visual domain.

## [Chapter 2](03_identifying/intro.md)
Given that some techniques have been presented that identify the parts of the data that are both sensitive and used to make a decision,
can we identify at deployment time, which individuals are most likely to have decisions influenced by a sensitive attribute?
And, in turn, can we use this to identify biased interpetation of features vs biased decisions?
Publications supporting this work are currently under review.

## Chapter 3

In the last chapter I look in general at methods to remove a sensitive attribute from data, whilst retaining other semantic features. 
In general this is a challenge. 
Most methods rely on significant compression of the input image. 
In this chapter I evaluate retaining as much sematic information as possible while also removing a sensitive attribute. 
I then suggest an additional approach based on Mixup.

## Appendix
- [Publications](09_appendix/published.md)
- [Software](09_appendix/software.md)
