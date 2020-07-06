# This Year

```{admonition} WIP
:class: Tip
This content is work in progress.
```

## 2019-2020: Full-Time

Since the last annual review, I worked on 3 papers

1. _[Imagined Examples](../09_appendix/imagined.md):_ We use a Variational Autoencoder model to produce a model that not only considers the features, 
but also the reported outcomes, as a source of bias. 
We use this model to generate likely counterfactual samples but without the need for a causal model.
2. _[Null-sampling for Invariant and Interpretable Representations](../09_appendix/nosinn.md):_ We use the bijective property of invertible neural networks in conjunction with an adversarial network 
to _disentangle_ a sensitive attribute's effect on other features. 
We then generate samples that are neutral with respect to a sensitive attribute, using these neutral images as our 
representation of the data we can investigate how the model has made the data "fairer".
3. _[Invisible Demographics](../09_appendix/invisible.md):_ Training data doesn't always reflect the deployment setting, with some demographic groups 
either mis-represented, or simply missing. 
We use unlabelled, but representative, data to guide our learned representations.

In addition I managed the Surgo reseach project for 7 months to produce the causal modelling tool [Intervene](https://github.com/predictive-analytics-lab/Intervene).

### Imagined Examples
Full text can be found in the [appendix](../09_appendix/imagined.md).

Learning a fair representation of data is a popular approach to mitigate unfair decisions by addressing a root problem
of learning from biased data.

However, almost every approach considers that the class labels are infallible.
The assumption is that bias exists only in the features our model takes as input.
This is however, not the case. 
Proxy labels are a source of bias and are not addressed in previous works.
 


### Intervene
As part of the Surgo research project we created the tool Intervene. 
This tool has 3 core functionalities:
1. Creating data
2. Fitting a Casual Model to data.
3. Evaluating the performance of the learned model.

#### Causal Modelling
Whilst the topic of causality is a research topic in it's own rite, there are clearly parallels between this area and 
the broad category of my research area - algorithmic fairness, accountability and transparency.

If we are able to accuractely model the causal relationships within the data, then we can ask fundamental questions,
interrogating the underlying frameworks that introduce bias into pur observations.
However, **accurate** causal modelling is not a solved problem.
My hope when beginning this research project was to  

#### Management
Falls outside of my research area, but happy to talk about this experience on request.  





## Other works

Gave a [presentation](https://predictive-analytics-lab.github.io/presentations/toronto2019.html#/) on fairness in machine learning at [Conference on Data Science and Optimization](http://www.fields.utoronto.ca/talks/Transparency-fairness) in Toronto.
