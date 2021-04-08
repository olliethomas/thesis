# Definitions of fairness

```{warning}
This content is very much under construction. This text should be considered PRE-first draft.
```

Fairness is a hard thing to define as it is contextual.
It is a word that is often used, without strictly thinking about its meaning.
Sometimes it’s easier to say that something is unfair, and fairness is assumed in the absence of unfairness.
This hasn’t stopped people trying to define it however, and the community has determined that there are two main notions of fairness, group fairness and individual fairness.
Group fairness is a popular definition as it is a statistical measure looking at the effect of on subgroups within a population.
Membership of these subgroups being defined by a member of the population’s sensitive attribute.
A sensitive attribute is an attribute that isn’t necessarily private, it might be quite clear to an observer, but it is an attribute that is protected by law.
Generally these are attributes that are inherent to an individual, something they are born with and have no choice over.
Examples include sex and race, though not every sensitive attribute falls into this definition (for example religion is a protected by law).
Group fairness definitions seek to enforce fairness by making sure that, on the whole, the members of one group are not more, or less disadvantaged than other groups.
There is particularly a concern that the majority group benefit more than minority groups.
One appeal is that this definition of fairness is relatively easy to calculate, providing that an observer has access to an individual’s group membership, their sensitive attributes.
Another appeal of this definition is that it is ingrained in law.
Particularly U.S. law, the 80% rule applies.
It would be remiss to not take into account the bearing that this has on U.S. produced systems.

An alternative definition of fairness is Individual Fairness.
This is an intuitive definition that similar individuals should be treated similarly to each other.
The problem arises however, that defining similarity in both input space and output space is a challenge.
A popular realisation of this definition is Counterfactual Fairness.
This asks the question, if an applicant were the same, except their sensitive attribute were intervened on to produce a counterfactual version of themselves, would they have the same outcome?
All of these are expanded on in later sections.
