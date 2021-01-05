# Imagined Examples for Fairness: Sampling Bias and Proxy Labels

Authors: O. Thomas, N. Quadrianto

---

```{math}

\newcommand{\kl}{D_{\text{KL}}}
\newcommand{\N}{\mathcal{N}}
\newcommand{\B}{\mathcal{B}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\xb}{\bar{x}}
\newcommand{\xbb}{\mathbf{\bar{x}}}
\newcommand{\yb}{\bar{y}}
\newcommand{\ybb}{\mathbf{\bar{y}}}
\newcommand{\0}{\mathbf{0}}
\newcommand{\I}{\mathbf{I}}
\newcommand{\x}{\mathbf{x}}
\newcommand{\s}{\mathbf{s}}
\newcommand{\y}{\mathbf{y}}
\newcommand{\zx}{\mathbf{z_x}}
\newcommand{\zy}{\mathbf{z_y}}
\newcommand\myeq{\mkern1.5mu{=}\mkern1.5mu}


```

## Abstract

Group notions of fairness such as demographic parity and equality of opportunity have been the focus of a growing body of work.
However, the intuitive notion of individual fairness, that similar individuals should be treated similarly has received less attention given the inherent obstacles in determining similar individuals.
We use generative modelling to produce an _imagined_ version for each individual example, to expand our dataset during downstream training.
Unlike other approaches, we explicitly view the reported labels, as well as the input features, to be a source bias.
This models known causes of unfairness in datasets, sample bias and proxy labels.
These imagined examples are produced by intervening on the protected characteristic and observing the affect on both the features and labels.
Experiments on four common fairness datasets show that by augmenting the entire training set with our generated interventions, we can identify and address the underlying cause of bias.

## Introduction

We address the problem of _unfair_ behaviour in machine learning decision systems.
Such decision system tasks include determining whether an applicant is likely to commit a crime while on bail, or whether a loan will be likely be repaid, amongst others.
The research community has responded to concerns that otherwise similar people, different only in a protected characteristic, are not treated similarly {cite}`Bolukbasi2016nlpbias,angwin2016machinebias`.
This has led to a growing body of work influenced by inter-disciplinary scholars to understand what is required to satisfy legal and ethical concerns about the deployment of machine learning models that reason about human behaviour.

It is generally understood that bias in machine learning systems arise as a reflection of the underlying biases in the training data.
This occurs when the training data implies a relationship between a protected characteristic, which we denote as $S$ and the observed outcome, $Y$.
That is, within the training dataset, the class label is not independent of a protected characteristic, $P(Y,S) \neq P(Y)P(S)$.

A simple and successful approach to redressing this problem is to re-weigh our training dataset so that this is no longer the case {cite}`kamiran2012data`.
This technique has, as a by-product, a useful property being that, in creating independence between $Y$ and $S$, two at-odds notions of group fairness, Demographic Parity (DP) and Equality of Opportunity (EqOp) can both hold.
Demographic Parity requires that the probability of a positive prediction be equal across all groups that share a protected characteristic.
In the case that both class label and protected characteristic are binary attributes, for example a loan approval system with a binarised gender value as the protected characteristic, so $Y=\mathrm{Approved}(1)$ or $Y=\lnot\mathrm{Approved}(0)$ and $S=\mathrm{Female}(1)$ or $S=\lnot\mathrm{Female}(0)$, DP would require that the prediction of the class label $\hat{Y}$ satisfy the following
\begin{equation*}
P(\hat{Y}=1 | S=1) = P(\hat{Y}=1 | S=0)
\end{equation*}

EqOp is similar to DP, but the probability of a positive prediction is also conditioned on the class label being positive.
This is equivalent to balancing the True Positive Rate across protected characteristic sub-groups
\begin{equation*}
P(\hat{Y}=1 | S=1, Y=1) = P(\hat{Y}=1 | S=0, Y=1)
\end{equation*}

It has recently been shown by Kleinberg et al. {cite}`Kleinberg2017tradeoff` and Chouldechova et al. {cite}`Chouldechova2017tradeoff` that unless certain conditions hold (we have a perfect dataset, the base acceptance rate is the same for both groups, i.e. class label is independent of the protected characteristic), demographic parity and equal opportunity cannot both hold.
Fortunately, by enforcing that the dataset has independence between the class label and the protected characteristic, both notions of fairness can be achieved.

While this is a welcome improvement, it has been proposed that this does not go far enough.
Recent works in Counterfactual Fairness {cite}`kusner2017counterfactual,Russell2017collide` propose that notions of fairness that seek to balance outcomes across sub-groups are not strict enough and that we should instead view fairness at an individual level.
Counterfactual fairness is a form of individual fairness based on causal inference.
Individual fairness requires that similar individuals be treated similarly.
This is difficult to quantify as the notion of 'similar' is not a rigid definition. Counterfactual fairness assumes we have a causal model that can generate our data samples. We can then generate a counterfactual example of a data sample obtained by amending a protected characteristic to another (valid) value. This is akin to asking the question, "What if I had been born another race?", or in the context of the loan decision example, "Would the same decision have been made if I had been born of a different gender?"
This can be modelled as

\begin{equation*}
P(\hat{Y}=1 | C(x, S=0)) = P(\hat{Y}=1 | C(x, S=1))
\end{equation*}

where $C$ is a function that generates the counterfactual example of an individual $X$ with a specified protected characteristic.

Obtaining causal models however, is difficult and expensive.
Furthermore, as we rarely have access to the ground truth causal model, there is often uncertainty among domain experts as to which causal model best characterises the data.
Recent work advocates taking samples from multiple causal models to obtain the most likely effect {cite}`Russell2017collide`.
Concurrently, works in Neural Processes have successfully emulated the behaviour of stochastic processes {cite}`garnelo2018neural,louizos2019functional`.
We draw on this and propose a stochastic process that emulates the behaviour of the counterfactual intervention function $C(\cdot)$.
That is, we propose a function that emulates a causal model that can only be intervened on with regard to a specific, pre-determined variable.
We set the variable which we will condition on to be the protected characteristic and observe the affect on the reconstructed counterfactual.\footnote{Code to reproduce results will be available}

Before we can remedy the problem however, we must understand how unfairness came to be included in our data.
Work to identify these sources has been conducted by {cite}`Tol19` who identifies that two of the ways bias can form in the data are either _sampling bias_ or, _proxy labels_.
Sampling bias occurs when the data is not representative of the population at large.
It may be that that task is legitimate, but there are not enough samples of particular subgroups in the data.
This can particularly be a concern when addressing intersectional fairness, when the numbers of individuals can be particularly small.
Proxy labels are used when the direct task label is not available and instead an alternative class label is given as a proxy for the outcome.
Unfairness occurs when this label is correlated with a particular subgroup.
The example of this is in predictive policing, where the data on which individuals have committed a crime is unavailable.
Instead, the closest we have is the details of those who have charged on suspicion of committing a crime.
If this proxy class label leads to a higher share of one protected group over another being targeted for arrest then we say the proxy label is biased.

This idea that bias may exist in the class label is corroborated by recent work by {cite}`kehrenberg2018interpretable,yeom` identifying that class labels themselves can be a source of bias.
This is in addition to work which seeks to remedy fairness by intervening on the features to create fair representations {cite}`edwards2015censoring,adel2019one,vfae,zemel2013learning`.
Our approach is to acknowledge that both the class labels and the features are a potential source of bias and intervene to identify and combat this bias.

In terms of a counterfactual model, if we had a sufficiently representative causal model, sampling bias can be identified by intervening on the causal model that recreates the features.
Identifying proxy labels can be identified by intervening in a causal model and observing the effect on the outcome.
If the intervention on the feature causes little to no change, then we note that the dataset is fair with regard to sampling bias.
If the intervention on the class label causes little to no change, then we note that the dataset is fair with regard to proxy label bias.

## Related Work

Most related to our work is the reweighing approach of Kamiran and Calders {cite}`kamiran2012data` which applies an instance weight for every group (Y,S) $\forall y \in Y$ and $s \in S$ so that the group is scaled to match the expected proportion of the groups in the data.
Doing so downweights members of the over-represented group while simultaneously upweighting those of under-represented groups.
Consequently, fairness criteria are improved without explicitly optimising for fairness.
This is similar to our approach in that we are not explicitly optimising to improve the fairness of our model.
Instead, we are trying to characterise and account for different types of bias within our data.
In doing so we aim to address the underlying cause of bias manifesting in the data as opposed to enforcing fairness on a biased dataset.

Our approach to _"imagining"_ is similar to that of Generative Adversarial Networks (GANs), which have achieved successes in a number of applications. Given their success there have been a number of variations of the framework. One popular approach {cite}`edwards2015censoring`is to make a latent embedding invarient to a protected characteristic and use this as the model input for a downstream task. Whilst this approach is successful in removing bias on the input features, bias in the class labels is not accounted for. In addition, recent works by {cite}`quadrianto2019discovering` demonstrate the inherent _uninterpretability_ of latent embeddings.
By reconstructing our imagined examples in the original input space we are able to determine which features are most connected to the protected charcteristic.

Counterfactual models have been the basis of a number of investigations into fairness {cite}`kusner2017counterfactual,nabi2018fair,chiappa2019path,KilBalKusWeletal19`.
This work is extremely encouraging, however requires specific application domain expertise in creating a causal model in which to intervene.
We take inspiration from this research area and try to emulate a subset of the behaviour of these counterfactual models.

## Imagined Examples Model

```{figure} assets/imagined/gm.png
---
height: 250px
name: gcm
---
**Left**: Graphical Model of our approach. **Right**: Our model in practice.
```

```{note}
In practice, we split the effect of $s$ into two intervention points. This allows us to condition the reconstruction of both the features $\x$ and class labels $\y$ individually. In the case that the features are invariant to the protected characteristic, the reconstruction will `ignore' the conditioning protected characteristic as $\zx$ will be sufficient for reconstruction. Similarly, if the class label is invariant to the protected characteristic then conditioning the class label decoder on the protected characteristic will not effect the reconstruction as sufficient information will be retained in $\zy$.
```

```{figure} assets/imagined/resampling_a.png
---
height: 250px
name: resampling_a
---
Removing sampling bias as the source of unfairness.
```

```{figure} assets/imagined/resampling_b.png
---
height: 250px
name: resampling_b
---
Removing proxy labels as the source of unfairness
```

```{note}
How do imagined examples work? In Figure {numref}`resampling_a`, we intervene on the protected characteristic and
observe the effect on the features producing $X^{\text{Imagined}}$ (see Figure {numref}`recons` for a visualisation of
this effect).
We can then use these imagined examples to augment the original dataset creating a balanced dataset in terms of group
proportions.
Since the imagined examples modify several features of the original examples, there will be several
_inconsistent features_ in the dataset, which will be ignored by the classifier.
Note that the procedure does not change the class label proportions.
In Figure {numref}`resampling_b`, we intervene on the proected characteristic and observe the effect on the class labels
producing $Y^{\text{Imagined}}$.
Here, the class label proportions will change.
Since the imagined examples copy the exact features of the original examples, there will be several _inconsistent
training examples_ in the dataset, which will be ignored by the classifier.
```

While prior works have focused on recreating the protected characteristic solely from the input features, recent work has also shown that the class labels themselves can be a source of bias {cite}`yeom,Tol19`.
In this context, if we had a set of features that were completely invariant to the protected characteristic, the model would either perform poorly, or have to learn to be biased to accommodate the biased labels.
To account for this we extend the graphical model of the unsupervised Variational Fair Autoencoder {cite}`vfae` to acknowledge that bias can also exist within the class labels (see figure {numref}`gcm` --- left).
In doing this, we explicitly model two latent embeddings that are disentangled from the protected characteristic for both the features and the decision outcome.
We refer to these latent representations as $\zx$ and $\zy$; while they could be modelled in the same space as the respective observed data, that need not be the case.

In practice however if we generate samples of $\x$ and $\y$, where both have been intervened on by the same variable, this is akin to na\"ively upsampling the underrepresented groups.
This causes over-representation of the minority groups and their observed outcomes and only exacerbates existing biases {cite}`kamiran2012data`. Instead, we 'split' the protected characteristic into two independent variables during decoding (Fig. {numref}`gcm` --- right).
One variable ($\s_1$) affects the reconstruction of the features and another ($\s_2$) that affects the reconstruction of the class labels.
However, during encoding we only consider the observed protected characteristic.

This allows us to intervene separately, either remedying the problem of under-represented samples in the data, or diagnosing that a task is not suitable to be learned given the data.

This generative process can be formally defined as

```{math}
\begin{align*}
&\mathbf{z_x} \sim p(\mathbf{z_X}); & \mathbf{x} \sim p_{\theta x}(\mathbf{x} | \mathbf{z_x}, \mathbf{s_1}); \\
&\mathbf{z_y} \sim p(\mathbf{\mathbf{z_Y}|\x}); & \mathbf{y} \sim p_{\theta y}(\mathbf{y} | \mathbf{z_y}, \mathbf{s_2}) \\
\end{align*}
```

where $p_{\theta x}(\mathbf{x} | \mathbf{z_x}, \mathbf{s})$ and $p_{\theta y}(\mathbf{y} | \mathbf{z_y}, \mathbf{s})$ are distributions that reflect the data modelled.
As $\s_1$ and $\s_2$ are marginally independent of $\mathbf{z_x}$ and $\mathbf{z_y}$ respectively we follow {cite}`vfae` and cast the problem of finding an invariant representation as performing inference on the graphical model and obtaining the posterior distributions of $\zx$ and $\zy$ by $p(\zx|\x,\s)$ and $p(\zy|\x,\s)$.

We parameterize the generative models (decoders) $p_{\theta_x} (\x|\zx, \s_1)$ and $p_{\theta_y} (\y|\zy, \s_2)$ and the variational posteriors (encoders) $q_{\phi_x} (\zx|\x, \s)$ and $q_{\phi_y} (\zy|\x)$ with neural networks, giving the following lower bound

```{math}
\begin{align}
\begin{split}
\sum_{n=1}^N &\log p(x_n|s_n) \geq \\
\sum_{n=1}^{N} & \kl(q_{\phi x} (\zx_n|\mathbf{x}\_n,\mathbf{s}\_n) || p (\zx_n)) \\
+& \kl(q_{\phi y} (\zy_n|\mathbf{x}\_n) ||p (\zy_n)) \\
-& \log q_{\phi y} (\y_n|\x_n) \\
+& \mathbb{E}_{q_\phi(\zy_n, \zx_n|\x_n, \s_n)}[- \log p_{\theta x} (\x_n|\zx_n, \s_{1n}) \\
+&\kl(p_{\theta y} (\mathbf{y}\_n|\mathbf{x}\_n) || q_{\phi y) (\mathbf{y}|\zy_n, \mathbf{s}\_n)} \\
-& \log p_{\theta y} (\y_n|\zy_n, \s_n)] \\
= &\mathcal{F}(\theta, \phi, \x_n, \s_n)
\end{split}
\end{align}
```

We assume the posterior $q(\zx, \zy|\x, \s, \y)$ factorizes to $\frac{q(\zx|\x,\s)q(\zy|\x)q(\y|\zy,\s)}{q(\y|\x,\s)}$, where

```{math}
\begin{align*}
p(\zx) \triangleq &\: \N(\zx|\0, \I) \\
p*\theta(\x|\zx, \s) \triangleq &\: f*\theta(\zx, \s) \\
p*\theta(\zy|\x) \triangleq &\: \mathrm{Cat}(\zy|\pi*\theta(\x)) \\
p*\theta(\y|\zy, \s) \triangleq &\: \mathrm{Cat}(\y | \pi*\phi(\zy, \s)) \\
q(\zy) \triangleq &\: \mathcal{U}(\zy) \\
q*\phi(\zx|\x, \s) \triangleq &\: \N(\zx|\mu*\phi(\x, \s), \sigma*\phi(\x, \s)) \\
p*\theta(\y|\x) \triangleq &\: \mathrm{Cat}(\y|\pi\_\phi(\x))
\end{align*}
```

where $f_\theta(\zx, \s)$ is a distribution suited to the data.
% Practically, $\mathrm{Cat}(\ybb| \y, \s)$ can be modelled with a Categorical distribution that is uniformly certain about all possible values.
To encourage $\zx$ and $\zy$ to be invariant to $\s$, we use adversarial network heads to predict the sensitive attribute from the latent distributions.
These are trained using the gradient reversal layer (GRL) of Domain Adversarial Neural Networks {cite}`ganin2016domain`, such that a min-max game is played out between the adversarial heads and the encoders.

This produces a $\zx$ and $\zy$ that are invariant to $\s$, yet retain as much information as possible in order to be useful in reconstructing $\x$ and $\y$, respectively.
To accommodate the representation being invariant to the protected characteristic, during reconstruction we additionally supply the protected characteristic label to the decoder.
This allows the decoder to be as accurate as possible and allows the encoder to remove more information regarding $\s$ as this information is added later.
When reconstructing our data, we can then "test" how sensitive our reconstructions are to the protected characteristic.
Reconstructing $\x$ on a dataset that has little correlation between the features and the protected characteristic will not result in the loss of much information information in the transformation to $\zx$, while decoding will also not be reliant on conditioning the decoder on $\s$.

(table)=

````{tabs}

```{tab} UCI Adult Dataset

| Strategy          | Accuracy $\uparrow$ | DP $\downarrow$    | EqOp $\downarrow$  | Ind.DP $\downarrow$ | Ind.EqOp $\downarrow$ |
| :---------------: | :-----------------: | :----------------: | :----------------: | :-----------------: | :-------------------: |
| No Intervention   | $83.909 \pm 0.311$  | $18.278 \pm 0.787$ | $13.518 \pm 3.481$ | $16.270 \pm 0.473$  | $30.918 \pm 1.585$    |
| Reweighing        | $83.409 \pm 0.138$  | $9.219 \pm 0.513$  | $10.019 \pm 2.386$ | $18.330 \pm 0.584$  | $32.344 \pm 0.923$    |
| Intervene on $X$  | $83.318 \pm 0.253$  | $12.498 \pm 0.258$ | $5.312 \pm 2.603$  | $11.446 \pm 0.294$  | $23.530 \pm 0.271$    |
| Intervene on $Y$  | $82.351 \pm 0.384$  | $2.771 \pm 0.856$  | $22.276 \pm 1.996$ | \-\-\-              | \-\-\-                |
| Augment with both | $82.131 \pm 0.512$  | $4.120 \pm 1.027$  | $17.294 \pm 2.689$ | $14.159 \pm 0.246$  | $28.336 \pm 0.633$    |
```
```{tab} German Credit

| Strategy          | Accuracy $\uparrow$ | DP $\downarrow$    | EqOp $\downarrow$  | Ind.DP $\downarrow$ | Ind.EqOp $\downarrow$ |
| :---------------: | :-----------------: | :----------------: | :----------------: | :-----------------: | :-------------------: |
| No Intervention   | $73.293 \pm 2.731$  | $4.429 \pm 5.307$  | $8.445 \pm 8.095$  | $28.743 \pm 4.888$  | $43.983 \pm 4.082$    |
| Reweighing        | $73.772 \pm 2.624$  | $6.344 \pm 6.170$  | $11.434 \pm 7.799$ | $28.323 \pm 3.689$  | $43.765 \pm 3.795$    |
| Intervene on $X$  | $74.551 \pm 2.117$  | $4.225 \pm 3.236$  | $5.560 \pm 3.497$  | $24.790 \pm 4.843$  | $43.632 \pm 4.514$    |
| Intervene on $Y$  | $73.293 \pm 1.764$  | $4.758 \pm 5.459$  | $7.445 \pm 7.204$  | \-\-\-              | \-\-\-                |
| Augment with both | $73.892 \pm 2.046$  | $5.060 \pm 3.413$  | $5.350 \pm 3.880$  | $26.707 \pm 4.145$  | $44.561 \pm 4.300$    |
```
```{tab} ProPublica COMPAS

| Strategy          | Accuracy $\uparrow$ | DP $\downarrow$    | EqOp $\downarrow$  | Ind.DP $\downarrow$ | Ind.EqOp $\downarrow$ |
| :---------------: | :-----------------: | :----------------: | :----------------: | :-----------------: | :-------------------: |
| No Intervention   | $67.004 \pm 0.717$  | $14.921 \pm 1.263$ | $16.000 \pm 2.018$ | $28.220 \pm 0.966$  | $34.238 \pm 1.723$    |
| Reweighing        | $66.819 \pm 0.569$  | $14.017 \pm 1.460$ | $15.044 \pm 2.248$ | $27.354 \pm 0.929$  | $34.013 \pm 2.284$    |
| Intervene on $X$  | $66.313 \pm 0.724$  | $15.306 \pm 0.475$ | $16.705 \pm 1.377$ | $24.300 \pm 0.667$  | $29.527 \pm 1.579$    |
| Intervene on $Y$  | $65.817 \pm 0.655$  | $11.321 \pm 3.365$ | $12.305 \pm 5.262$ | \-\-\-              | \-\-\-                |
| Augment with both | $66.313 \pm 0.921$  | $13.036 \pm 2.224$ | $14.624 \pm 3.590$ | $25.671 \pm 0.964$  | $32.044 \pm 2.229$    |
```
```{tab} NYC SQF

| Strategy          | Accuracy $\uparrow$ | DP $\downarrow$    | EqOp $\downarrow$  | Ind.DP $\downarrow$ | Ind.EqOp $\downarrow$ |
| :---------------: | :-----------------: | :----------------: | :----------------: | :-----------------: | :-------------------: |
| No Intervention   | $91.948 \pm 0.548$  | $0.980 \pm 0.613$  | $4.467 \pm 3.075$  | $6.103 \pm 1.878$   | $33.118 \pm 1.332$    |
| Reweighing        | $91.958 \pm 0.546$  | $0.961 \pm 0.578$  | $4.286 \pm 2.730$  | $5.865 \pm 1.111$   | $34.591 \pm 1.411$    |
| Intervene on $X$  | $91.939 \pm 0.598$  | $0.761 \pm 0.381$  | $4.129 \pm 2.522$  | $4.976 \pm 0.316$   | $26.733 \pm 1.652$    |
| Intervene on $Y$  | $92.065 \pm 0.458$  | $0.810 \pm 0.280$  | $3.912 \pm 1.931$  | \-\-\-              | \-\-\-                |
| Augment with both | $91.973 \pm 0.492$  | $0.729 \pm 0.334$  | $4.602 \pm 2.735$  | $5.952 \pm 0.327$   | $28.017 \pm 1.508$    |
```
````

```{figure} assets/imagined/diff_all.png
---
height: 250px
name: resampling_b
---
The first $100$ reconstructed samples of one random repeat from the UCI Adult Income dataset drawn $3$ times. Each image corresponds to the difference in reconstructions given a sample from $\zx$ and $S=s, \forall s\in \{0,1 \}$. Features that we do not believe to be related to the sensitive attribute _sex_, such as _race_ remain unchanged, while features we believe are strongly related to the sensitive attribute, such as the _relationship_ status attribute value _husband_, consistently change when the sensitive attribute is altered. Features that we are more uncertain about, such as highest attained _education_ level and _hours per week_ worked are more inconsistent in their behaviour.
```

However, if the features are entwined with the protected characteristic, then a significant amount of this information will be removed during the transformation to $\zx$.
The decoder network will then rely on the protected characteristic to reconstruct the features.
We exploit this relationship by reconstructing an imagined counterfactual by supplying a "flipped" sensitive attribute and augment our dataset with these additional examples.

In terms of the features, this is similar to asking in what ways you would likely be different if you were of (for example) another gender.
We also perform these interventions on the reconstruction of the class label, which equates to asking if the same outcome would have been observed if you were of (for example) another gender.

## Results

We evaluate the performance of $3$ augmentation strategies on $4$ commonly used fairness datasets.

**UCI Adult Income**{cite}`Asuncion+Newman:2007`. A dataset of 45,222 samples, from the 1994 U.S. census.
The binary classification task is predicting whether an individual's salary is greater than \$$50,000$ USD; the binary sensitive attribute is whether an individual's sex is Male or not.

**UCI German Credit**{cite}`Asuncion+Newman:2007`. A dataset of 1000 samples from the UCI Repository of Machine Learning Databases of data used to evaluate credit applications in Germany.
The binary classification task is predicting if an individual's credit label is positive or not; the binary sensitive attribute the individual's sex.

**COMPAS**. A dataset of 6,167 samples, released by ProPublica following their investigation into recidivism prediction.
The binary classification task is predicting if an individual is charged with an act of recidivism within $2$ years of being released, and the binary sensitive attribute is whether an individual's race is White or not.

**NYC SQF**. A dataset of 12,347 samples, from 2016 of data on individuals stopped as part of New York City's Stop, Question, Frisk initiative.
The binary classification task is predicting if a stopped individual is found to be carrying a weapon or not, and the binary sensitive attribute is whether an individual's race is White or not.

The augmentation strategies are **Intervene on $X$** The original data is augmented with examples that are produced by intervening on $\s_1$ during decoding. To emulate the behaviour of uncertainty over causal models {cite}`Russell2017collide` we draw $3$ samples from $\zx$ with which to condition our decoder. The dataset then contains the original data tuple $(\x, \s, \y)$ three times and three `imagined' examples $(\x_\mathrm{imagined}, \s_\mathrm{flip}, \y)$.
**Intervene on $Y$** We make a similar intervention, but on the class label reconstruction, augmenting the original dataset $(\x, \s, \y)$ with $(\x, \s_\mathrm{flip}, \y_\mathrm{imagined})$
**Augment with both** We make both of the previous interventions and augment the dataset with both so that the new dataset comprises of examples from $(\x, \s, \y)$, $(\x_\mathrm{imagined}, \s_\mathrm{flip}, \y)$ and $(\x, \s_\mathrm{flip}, \y_\mathrm{imagined})$.

All values in the dataset are one-hot encoded, including continuous values which are split into $5$ bins.
The exception to this is the ProPublica COMPAS Recidivism dataset where performance is largely determined by the continuous values,
In this dataset we scale the values to be in the range $[0, 1]$.
The data is split into a train and test set using a two-thirds, one-third split.
All results are reported on the same test set per repeat.
Dataset augmentation only occurs in the training set.
A cross-validated logistic regression model is used in each experiment.
We use $3$-fold cross-validation over a range of regularisation parameters $10^{-i}$ $\forall i \in [0, \dots, 6]$.
We repeat each experiment $5$ times, using a different seed to split the data for each repeat.

The value reported for DP is the absolute difference in the positive prediction rate between the protected characteristic subgroups.
Similarly, the value reported for EqOp is the absolute difference in the True Positive Rates across subgroups.
Ind. DP. is measured as the mean absolute difference in the probability of a positive prediction conditioned on $X$ generated with $S=0$ and $X$ generated when $S=1$.
\begin{equation*}
\lvert P(\hat{Y}=1 | f(Z_x, S_0)) - P(\hat{Y}=1 | f(Z_x, S_1)) \rvert
\end{equation*}
Ind. EqOp. is Ind. DP., but also conditioned on the observed class label being positive.

\begin{equation*}
\lvert P(\hat{Y}=1 | f(Z_x, S_0), Y\myeq{}1) - P(\hat{Y}=1 | f(Z_x, S_1), Y\myeq{}1) \rvert
\end{equation*}

We compare our model against an unconstrained Logistic Regression model, and one to which the instance weighting scheme of {cite}`kamiran2012data` is applied.
In addition we train a Logistic Regression model on our augmented samples.

In all experiments, to account for uncertainty in the reconstruction we draw three samples from the feature embedding space with which to condition our decoder.
An example of this is given is fig. {numref}`recons`.
For the prediction encoder we model the latent space as the `true' label prior to conditioning on the sensitive attribute, as such we only draw one sample as we use the probability directly.

The results of all experiments are shown in table {numref}`table`.
Across $4$ datasets we observe that the effect of intervening on either the reconstruction of the feature, or the class label has differing results.
We attribute this to two phenomenon, sampling bias and proxy labels.
In the case of the UCI Adult dataset and the German Credit dataset, the performance across (almost) all metrics is increased by intervening on the reconstruction of the _features_, including individual measures of fairness.
Intervening on the class label however, has either a negligible or erratic effect.
We conjecture that this is due to Adult Income and German Credit datasets suffer from sampling bias but not proxy labels problems.

Conversely, in the crime-related dataset, the ProPublica COMPAS Recidivism dataset and the NYC Stop, Question, Frisk dataset better results are obtained by intervening on the class label.
We attribute this to bias caused by proxy labels.
In both these datasets we are using a label to emulate another.
We use arrest data to determine if a crime has been committed, regardless of the fact that only those who have been stopped can be arrested.
Unfortunately, many crimes are not recorded as the perpetrator is not caught.

## Discussion and Conclusion

```{epigraph}
Imagine all the people, sharing all the world

-- John Lennon
```

We have used generative modelling to 'imagine' likely counterfactual examples. Unlike previous work, we explicitly consider both the features and the class label to potentially be a source of bias. We make separate interventions on a protected characteristic in both of these spaces and observe the effect of the intervention on reconstruction. We then augment our training set for a downstream task with these imagined examples. In the case that the data is not particularly representative of individual groups sharing a protected characteristic, this can be attributed to _sampling bias_. We propose that by`imagining' examples of the underrepresented group, we can improve the robustness of a classifier, giving fairer results in terms of group and individual fairness metrics for free despite not explicitly constraining a learning model for the downstream task.
This is achieved by providing examples of inconsistent features within the data.

In the case where the labels are a source of bias, we attribute this to an indirect _proxy label_ which is correlated with a sensitive attribute.
Intervening just on the class label and augmenting our training set with these labels tend to produce fair results by ignoring samples which are contradictory (inconsistent samples).

We perform experiments on four commonly used datasets in the fairness literature: Adult Income, German Credit, Propublica COMPAS, and NYC SQF. Our results point to the conclusion that the two financial-related datasets (Adult Income and German Credit) suffer from sampling bias, while the crime-related datasets (Propublica COMPAS and NYC SQF) suffer from proxy label.

As future work, we will study the interaction between sampling bias and proxy labels.
The results imply that the relationship between them is more complex than can be resolved simply by intervening on both the features and the class label at the same time, or jointly but separately.
We aim to characterise this relationship in greater detail, investigating if both inconsistent features and inconsistent samples can be handled together within our framework.

```{bibliography} ../../references.bib

```
