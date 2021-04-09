# An Algorithmic Framework for Positive Action

In previous work we have identified characteristics that are overly relied on and lead to unfair decisions.
This work amends this problem statement to instead of identifying characteristics and mitigating them, identify idividuals who are at risk of receivingan unfair decision.

## Abstract
We present an algorithmic framework to promote positive action towards diversity and equality. 
Positive action is designed to help individuals from under-represented groups to overcome systemic disadvantages. 
Measures may include a more diverse interview panel, targeted training, or adaptive policies. 
To achieve this, we present a hybrid fairness axiom that combines the ‘we are all equal’ and ‘what you see is what you get’ fairness axioms through adding a temporal component to the construct space. 
We demonstrate a learnt counterfactual decision algorithm, based on this hybrid axiom, that both diagnoses a number of biased behaviours, and identifies individuals who have the potential to benefit from positive action. 
Inspired by deferment, we extend beyond an accept / reject approach to include a third, ‘positive action’ option as a long-term bias mitigation strategy. 
Experiments on a synthetic and real-world datasets show that we can identify individuals most likely to suffer from unfair treatment, and discuss potential approaches to address this.

## Introduction
```{epigraph}
Can we employ AI to aid positive action towards equality & diversity?
```
As algorithmic decision making (ADM) systems are deployed in increasingly consequential settings, the pressure to guarantee ethical, or at least lawful, performance accelerates. 
Unchecked, an ADM system can display biased, discriminatory behaviour, caused by the reinforcement of existing human biases found within the training data, and even introduce unexpected new ones {cite}`BarSel16,BarCraShaWal17,ChoRot20`. 
The machine learning community has responded, producing a growing number of methods that aim to reduce, mitigate, or remove this unintended behaviour displayed by an ADM system {cite}`PedRugTur08,KamAkaAso2012,KamCal12,ZemWuSwePitDwo13,MadCrePitZem18`. 
However, there is no consensus regarding the best approach for bias removal in a real world setting.

Although non-discrimination is protected by law in the EU, U.S. and beyond, the legal definition of discrimination is contextual, and not well defined enough to be automated {cite}`Wachter20`. 
The notions of fairness and equality are perhaps even more elusive and can vary between definitions aimed at protecting individuals, or alternatively, protecting groups. 
Successfully incorporating an unbiased algorithmic decision maker, however, requires a sound understating of the organisation's equality \& diversity objectives in both the short and long term. 
Short-term goals may focus on aligning with legal requirements and ensuring a non-discriminatory process, and long-term term goals may include a desire to increase diversity or representation of under represented groups. 
While positive discrimination is generally discouraged and can even be unlawful within the EU, organisations may wish to take _positive action_; 
lawful measures taken to encourage and train people from under-represented groups to help them overcome disadvantages in competing with other applicants[^4]. 
Positive action examples include universities offering a foundation year to disadvantaged students and the Athena SWAN charter. 

We propose a computational framework and a counterfactual deferral model that promotes positive action towards equality in addition to aligning with non-discrimination legal requirements.

## Background
In this section, we first motivate the idea of causality and counterfactual modelling for detecting biased behaviour.
We then describe differing interpretations of fairness constraints under different axioms or worldviews.

### Counterfactual Modelling
To detect bias, we first need to evaluate the effect of the sensitive attribute on the outcome.
A sensitive attribute can be race, gender (protected characteristic) or any attribute we wish the outcome to be independent of. 
Ideally, we want to understand all the relationships between the measured attributes and the outcome. 
One way to achieve this is by employing a Structural Causal Model (SCM) - a graphical model consisting of a number of vertices which represent features, and edges, which represent the _causal_ pathway between them {cite}`Pearl09`. 
A complete, unique structural model, however, is application specific and requires verification by domain experts.

Another approach to understanding causality is through the notion of counterfactual or potential outcome: 
"event C is said to have caused event E if, under some hypothetical counterfactual case the event C did not occur, E would not have occurred" {cite}`Hume00, Miller19`.
In the context of bias and equality, we can ask the hypothetical question: 
if we could “flick a switch” and change the sensitive attribute of an individual, will the decision outcome change?
In practice, we can apply this principle by finding two as-close-as-possible individuals (differing by the sensitive attribute) within the data (e.g. {cite}`RosRub83,Rubin90`), or, recently, by creating the counterfactual representations by an adversarial learning model (e.g. {cite}`MadCrePitZem18b,ShaHenDarQua20,GoeGuLiRe20`).
The first matching approach provides inter-individual differences (_normative_) assessment, while the latter adversarial approach provides intra-individual differences (_ipsative_) assessment.
In this paper, we focus on the latter one.

### Fairness definitions and Axioms
The computational framework presented in this work mainly builds on that presented in {cite}`FriSchVen16` which defines the construct space, observed space and decision space, and suggested several formalised definitions of bias by utilising the mappings between these spaces.

The _construct space_ represents the ground truth -- an unobserved space that correctly captures closeness between individuals with respect to a task -- The _observed space_ represents the measurable features for consideration, and the _decision space_ the outcome {cite}`FriSchVen16`. 
For example, intelligence resides in the construct space, IQ, measured by an IQ test, resides in the observed space, and acceptance or rejection from the International Mensa club resides in the decision space.

As the observed space is an approximation of the construct space, we are required to make an assumption regarding the accuracy and reliability of this approximation; 
these assumptions are refereed to as axiom assumptions or 'worldviews'.
In the context of fairness, we can distinguish between two important axiom assumptions: WAE ("we're all equal") and WYSIWYG ("what you see is what you get").
WAE (Figure~\ref{fig:gcm_redux} -- top right) assumes decisions, if correctly based on the ground truth, will be independent of the sensitive attribute.
WYSIWYG (Figure~\ref{fig:gcm_redux} -- top left), on the other hand, allows for a disparity between protected groups, assuming the observed discrepancies are a true reflection of the ground truth.
These worldview assumptions are in tension, and considered to be fundamentally incompatible with each other {cite}`FriSchVen16,BluSta20`.

With these worldviews in mind, we can discuss two statistically measurable fairness definitions: Demographic parity (`DP`) and equality of opportunity (`EqOp`). Demographic parity, which requires the rate of positive outcome to be equal across protected groups, links with the WAE axiom {cite}`FriSchVen16,YeoTsc21`.
For example, if a bank approves a loan for 50\% of white applicants, under demographic parity, it will be required to to approve loans for 50\% of black applicants as well, regardless of their credit score.
Recently, {cite}`Wachter20` proposed conditional demographic (dis)parity (CDD)[^1] as a statistical baseline that aligns with the EU legislation.
CDD requires demographic parity to be enforced only to the point it can occur without positive discrimination.
In this example, demographic parity will be enforced only with the condition that the credit scores threshold is also being satisfied.
Equality of opportunity (\texttt{EqOp}) is another important group fairness definition.
EqOp requires we enforce the True Positive Rate (\texttt{TPR}) to be equal across protected groups and links with the WYSIWYG axiom {cite}`HarPriSre16,YeoTsc21`[^2].
Referring back to the loan example, under EqOp we will require the loan would be approved s.t. the same \% of black and white applicants actually go on to repay the loan.

To better understand the statistical disparity we may witness in the data, it is useful to discuss potential biases.
Focusing only on bias within the observed and decision spaces, we can categorise bias within data into two broad categories -- a misleading representation of the input attributes (such as selection bias) and a misleading set of outcomes (such as label bias).
_Sample selection bias_ originates from training on a non-representative sample of the population due to a systematic error in data collection {cite}`Tol19`.
_Label bias_ occurs when recorded outcomes within the data are discriminate between protected groups {cite}`WicPanTri19,JiaNac20`.
Mitigation efforts that consider selection bias {cite}`KamCal12`, or label bias {cite}`CalVer10,JiaNac20` independently are available.

Treatment of fairness in machine learning is often limited to the observed space and the decision space.
Thus, any effect of bias introduced outside of the environment which we can control -- training population, measurements, learning algorithm -- isn't taken into consideration. This can give the false hope that deploying a locally-fair AI decision-maker can alleviate bias in the environment it is deployed in {cite}`FeiGogUhl21`.
We aim to encourage positive action trough understanding a broader range of biases, including bias that is external to the ADM system and therefore cannot be genuinely mitigated by an automated rejection / acceptance model.

```latex

\begin{figure*}[ht]
    \centering
    \includegraphics[width=0.95\textwidth]{assets/figure1.pdf}
    \caption{%
    A graphical visualisation of the `hybrid' axiom and biases potentially introduced at each mapping step.
    The potential at birth is assumed to be independent of all protected characteristics.
    At the point of observation, however, disparity between sub-groups is allowed, due to the presence of `lifetime' bias.
    We define the WAE and the WYSIWYG axioms as separated by three steps, each with the possibility of bias being introduced.}%
    \label{fig:story}
\end{figure*}
```

### Contributions

Specifically, our paper provides the following main contributions:

1. A `hybrid' fairness axiom that combines both the WAE and WYSIWYG axioms, by further decomposing the construct space, to allow these axioms to co-exist[^3].
2. A `positive action candidate' (PAC) outcome, a decision that is neither accept/reject, to accommodate the suggested framework and promote positive action towards diversity \& equality
3. A counterfactual deferral model that allows flexible bias auditing and mitigation via a set of adaptable selection rules.

## Positive Action Framework
We first describe our proposed fairness axiom that will allow us to define a positive action candidate.

\subsection{A ‘hybrid’ fairness axiom}
The axiom adopted in this work is visually presented in Figure~\ref{fig:story}.
We expand the _construct space_ to include the element of time, with the observed space containing discrete measurements of the construct space in time.
Let's consider a measurable feature $X$ within the observed space; $X$ is an approximation to its non-measurable counterpart $\tilde{X}$. We assume that $\tilde{X}$ decomposes into the convex combination:
```latex
\begin{equation}
  X\approx\tilde{X}=\alpha\cdot\tilde{X}_{b}+\beta\cdot\Delta\tilde{X}
\end{equation}
```
where
```latex
\begin{align*}
  \tilde{X}_{b}\perp s \quad\text{and}\quad
  \Delta\tilde{X} \not \perp s~~,
\end{align*}
```
the sensitive attribute being $s$, and $\alpha$ and $\beta$ being non-negative values that sum to 1.
In words, we assume an individual’s suitability for the task, at the time of the measurement, is a combination of their aptitude ($\tilde{X}_b$) and their experiences over time ($\Delta \tilde{X}$)[^5].
We further assume that the 'aptitude' component, $\tilde{X}_{b}$, with regard most tasks[^6], is independent of any protected characteristic, and hence complies with the WAE axiom.
The 'life-experience' component $\Delta\tilde{X}$, that shifts the 'aptitude' either positively, or negatively, may not be independent of a protected characteristic.
$\tilde{X}$ represents the non-observable ground truth, per individual, at the time of measurement, and may no longer be independent from the protected characteristic $S$, due to bias experienced prior to the observation.
We define the effect $S$ has on $\Delta\tilde{X}$, which may be a collection of several influences, as `lifetime' bias.


However, there are many additional types of biases that can form within our data, prior to the collection of observed data. One example is self-selection bias, which may be present due to lack of awareness or lack of diverse role models, and can obscure the decision maker's view of the construct space.
When mapping from the construct space to the observed space, measurement bias can be introduced through the observation itself, as the observation may not reflect the potential equally well for under-represented groups.
And, the mapping from the observed space into the decision space may introduce further bias, such as label bias from the use of a proxy variable, and potentially, direct discrimination.
The assumption made in this work is that the observed data represents the best available success-predictor that can be employed under the circumstances, and that it provides, at a minimum, a reasonable predictive success across the sub-groups.
This way, despite our underlying assumption of the initial construct space being WAE, once the observed data is collected, WYSIWYG can be employed.

### Counterfactual Deferral Model

```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=0.3\textwidth]{assets/figure2_rev.pdf}
    \caption{%
    Graphical Model of our `hybrid' fairness axiom.
    The labelling $b_{1}$, $b_{2}$ and $b_{3}$, corresponds to the lifetime bias, measurement bias, and direct bias, respectively (see also Figure~\ref{fig:story}).}%
    \label{fig:our_model}
\end{figure}
```

#### Graphical model
A graphical model of our approach is shown in Figure `\ref{fig:our_model}`.
Corresponding to Figure `\ref{fig:story}`, the model is situated within the construct, observed and decision spaces.
As mentioned above, within the construct space $\tilde{X}_{b}$ represents the 'aptitude' component, whereas $\tilde{x}$ represent the non observable true potential, with respect to a task, at the point of measurement.
While $\tilde{X}_{b}$ is independent of $s$, $\tilde{x}$ contains an additional component $\Delta\tilde{x}$ that is not independent of $s$, hence $\tilde{x}$ is a child of $s$.
The observed space contains $x$, the observable attributes measured, and the decision space contains $y$, the output. 
Both $x$ and $y$ are assumed to be a children of $s$.
The labelling $b_{1}$, $b_{2}$ and $b_{3}$, defines the potential biases at each mapping, for example, lifetime bias, measurement bias, and direct bias, respectively, as shown if Figure `\ref{fig:story}`.

```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=0.45\textwidth]{assets/figure3.pdf}
    \caption{%
    Graphical representations of the joint probability of a population which are labelled with regard to reported outcome $Y$ and a protected characteristic $S$.
    \emph{Top row}:
    \emph{Left} -- In the WYSIWYG axiom the data is representative of a fundamental difference between protected groups.
    \emph{Right} In the WAE axiom, potential should be equally present across protected groups. Any mis-representation of this must be caused by 'lifetime' and measurement bias.
    \emph{Bottom}: Overlapping the above two axioms.
    The population captured by groups G1, G2, G5 and G6 are represented consistently.
    Groups G3 and G4 represent individuals that will receive different outcome if a different axiom is assumed.}%
    \label{fig:gcm_redux}
\end{figure}
```

```latex
\begin{figure*}[ht]
    \centering
    \includegraphics[width=0.8\textwidth]{assets/figure4.pdf}
    \caption{%
    Diagram representing our method.
    The original representation $x$ is mapped into a representation $z$ that is independent of the protected characteristic $s$.
    $z$ is mapped back into $x_{s_x=0}$ and $x_{s_x=1}$, reintroducing $b_{1}$ and $b_{2}$.
    Each of those representations are then labelled, reintroducing $b_{3}$, resulting in four representation in total.
    The four corresponding predicted outcomes then determine the group classification (according to Table \ref{table:1}) and one of three final outcomes: accept, reject, or flag for positive action.}%
    \label{fig:wae_wys_thirdoption}
\end{figure*}
```

### Counterfactual Approach

#### Splitting WAE and WYSIWYG in the middle

As a prelude to our counterfactual model, it is helpful to divide the data into six subgroups, as shown in Figure `\ref{fig:gcm_redux}`.
There groups are defined by overlaying the observed data, assuming a WYSIWYG axiom, (Figure `\ref{fig:gcm_redux}`, top left) on a representation of the data with demographic parity enforced, assuming a WAE axiom (Figure `\ref{fig:gcm_redux}`, top right).
When overlaid, the data can be separated into six subgroups, as shown in Figure `\ref{fig:gcm_redux}`, bottom.
Subgroup $G_{1}$ and $G_{2}$, represent a positive outcome under both axioms.
Subgroup $G_{5}$ and $G_{6}$, represent a negative outcome under both axioms.
Subgroups $G_{3}$ and $G_{4}$, however, represent _a different outcome under the two axiom_ (i.e. the two axioms collide).
Subgroup $G_{3}$, specifically, represents the subgroup that would have received a positive outcome if demographic parity would have been employed, and a negative outcome based on WYSIWYG.
If following our hybrid axiom, this subgroup can be interpreted as individuals with potential, which may have been negatively impacted by 'lifetime' bias. 
We note that giving this group an immediate positive outcome may not be helpful or practical. However, they may benefit from positive action that may later lead to a positive outcome.

In this work we demonstrate using counterfactual modelling to create a group classifier that both mitigates direct bias and identifies the members of sub-group $G_{3}$.
For simplicity, we provide illustrations for a binary protected attribute, the proposed approach generalises to a multi-level attribute.

#### Building a group classifier.
We use a general two-step approach following the scheme in Figure `\ref{fig:wae_wys_thirdoption}`.
First, we train an adversarial auto-encoder model that performs translations within the domain of the protected attribute: the observed data $x$ is mapped into a representation $z$, that is independent of the protected characteristic.
From $z$, two mirror representations can be created, $x_{s_x=0}$ and $x_{s_x=1}$. 
The variables $x_{s_x=0}$ and $x_{s_x=1}$ are labelled to create four representation in total: $x_{s_x=0,s_y=0}$, $x_{s_x=0,s_y=1}$, $x_{s_x=1,s_y=0}$ and $x_{s_x=1,s_y=1}$. 
Here, $S_y$ denotes the direct label attached to the set of attributes.
For the second step, we train a second model, a classifier ('decision predictor' in Figure `\ref{fig:wae_wys_thirdoption}`) to performs predictions on the counterfactual representations. 
When we feed in the counterfactuals, we get a corresponding set of outputs:
$y_{s_x=0,s_y=0}$, $y_{s_x=0,s_y=1}$, $y_{s_x=1,s_y=0}$ and $y_{s_x=1,s_x=1}$.
With this knowledge, the group classifier then sorts the original data representation $x$ into one of the subgroups $G_{1-6}$.
Subgroups $G_{1,2,4}$ receive a positive outcome, subgroups $G_{5,6}$ receive a negative outcome, and members of subgroup $G_{3}$ are highlighted as _positive action candidates_.

Through this process we are able to separate biases that occur prior to our decision-making system, and those that arise from within it.
To better understand the difference between $S_x$ and $S_y$, consider the following illustration:
A titled chess player ($CP$) from Krakozhia, is ranked within the top $5\%$ of players in her country.
In this example, $s$ is country of origin and $y$ is whether or not an invitation to an international tournament is received.
To determine if the process for receiving a tournament invitation is fair, we can
follow the process shown in Figure `\ref{fig:wae_wys_thirdoption}` and create a counterfactual player, from rival nation Syldavia, such that $CP_{s_x=1}$ is our chess player having grown with all the benefits of being a Krakozhian player, and placing in the top $5\%$ of her country, and $CP_{s_x=0}$ is our chess players, but with the best guess of how their chess skill would have progressed if raised in Syldavia, placing her in the top $5\%$ in her country.
$S_y$, in this case, is the country of origin, as stated in the application form.
For each player we create two tournament applications, resulting in four applications in total as shown in Table `\ref{table:chess}`:
$CP_{s_x=0,s_y=0}$, Syldavian player, Syldavia on the form; $CP_{s_x=0,s_y=1}$ Syldavian player, Krakozhia on the form; $CP_{s_x=1,s_y=0}$, Krakozhian player, Syldavia on the form; $CP_{s_x=1,s_y=1}$ Krakozhian player, Krakozhia on the form.

```latex
\begin{table}[h!]
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}rll@{}} \toprule
\multicolumn{1}{l}{} & \multicolumn{2}{c}{Country of origin on application form} \\
\cmidrule{2-3}
Country raised in & \multicolumn{1}{c}{Krakozhia ($S_y=1$)} & \multicolumn{1}{c}{Syldavia ($S_y=0$)} \\
\midrule
Krakozhia ($S_x=1$) & Invited?$_{s_x=1,s_y=1}$ & Invited?$_{s_x=1,s_y=0}$ \\
Sylvadia ($S_x=0$) & Invited?$_{s_x=0,s_y=1}$ & Invited?$_{s_x=0,s_y=0}$ \\
\bottomrule
\end{tabular}
}
\caption{Intervening on the protected characteristic at two stages of the decision-forming process allows for more precise diagnosis and remedy on undesired behaviour.}
\label{table:chess}
\end{table}
```

We then `submit' these application to the decision-predictor and receive four corresponding decisions.
The tournament invitation system can then be interrogated.
Importantly, unfair patterns resulting from the decision system, as opposed to development of the player, can be identified.

The group classifier takes the four outcomes predicted by the classifier and sorts to the appropriate subgroup, following a set of selection rules.
These group classifier rules are presented in Table `\ref{table:1}`.
When consensus is achieved between all counterfactual outputs, no bias is detected and the outcome is reported as dictated by the consensus.
If consensus is not achieved, the following rules apply (assuming bias is in favour of $s=1$):
```latex
\begin{itemize}
    \item If $y_{s_x=1,s_y=1}=1|s=0$ and \\ $\forall j \in \{0,1\}$ $y_{s_x=0,s_y=j}=0|s=0$ then $y=2$ \\ (we use $y=2$ to denote the outcome for positive action candidates).
    \item else, if $y_{s_x=1,s_y=1}=1$ then $y=1$.
    % \item If any $y_{s_x=1,s_y=i}=1$ and all $y_{s_x=0,s_y=i}=0$ and $s=1$ the output is $y=1$.
    % \item If any $y_{s_x=1,s_y=i}=1$ and all $y_{s_x=0,s_y=i}=0$ and $s=0$ the output is $y=2$ (we use $y=2$ to denote the outcome for positive action candidates)
    \item If $\forall i,j \in \{0,1\}$ such that $y_{s_x=i,s_y=j}=0$ then $y=0$.
\end{itemize}
```
Rules for bias in the opposite direction can be easily derived.

```latex
\begin{table*}[ht]
\centering
% \resizebox{\textwidth}{!}{%
\begin{tabular}{ @{}ccccccccc@{} }
 \toprule
   Selection Rule & $s$ & $y_{s_x=0,s_y=0}$ & $y_{s_x=0,s_y=1}$ & {$y_{s_x=1,s_y=0}$} & $y_{s_x=1,s_y=1}$ & Subgroup & Bias & $y$ \\ [0.5ex]
\midrule
 1 & 0 or 1 & 1 & 1 & 1 & 1 & $G_{1}$ or $G_{2}$ & - & 1\\
 2 & 0 or 1 & 0 & 1 & 1 & 1 &  $G_{1}$ or $G_{2}$ & $b_{3}$ & 1\\
 3 & 1 & 0 & 0 & 1 & 1 & $G_{4}$ & $b_{1}$ or $b_{2}$ & 1\\
 4 & 1 & 0 & 0 & 0 & 1  & $G_{4}$ & $b_{3}$ & 1\\
 5 & 1 & 0 & 1 & 0 & 1  & $G_{4}$ & $b_{3}$ & 1\\
 6 & 0 & 0 & 0 & 1 & 1 & $G_{3}$ & $b_{1}$ or $b_{2}$ & 2\\
 7 & 0 & 0 & 0 & 0 & 1  & $G_{3}$ & $b_{3}$ & 2\\
 8 & 0 & 0 & 1 & 0 & 1  & $G_{1}$ & $b_{3}$ & 1\\
 9 & 0 or 1 & 0 & 0 & 0 & 0 & $G_{5}$ or $G_{6}$ & - & 0\\
 \bottomrule
\end{tabular}%
% }
\caption{%
Selection rules for mapping from the groups represented in Figure~\ref{fig:gcm_redux} and Figure~\ref{fig:wae_wys_thirdoption} to a decision.
As $s=0$ represents an disadvantaged group, we identify those in group 3 to be suitable for \emph{positive action}. These are candidates who would have been accepted had a counterfactual version of themselves been considered.
N.b. Combinations not listed are identified and the outcome reverts to the outcome from an unconstrained model.}%
\label{table:1}
\end{table*}
```

## Implementation
Our model is implemented as two successive neural networks representing distinct phases as mentioned above.
Our goal in each model is to produce a counterfactual with respect to the protected characteristic $s$.
First, we aim to train an autoencoder model capable of producing a counterfactual _reconstruction_ in $\mathcal{X}$; and then we aim to train a classification model capable of producing a counterfactual _decision_ in $\mathcal{Y}$.


The adversarial autoencoder has a similar architecture to {cite}`MadCrePitZem18`, but with multiple decoders, and comprises of:
\begin{enumerate}
    \item An Encoder function $g: (\mathcal{X}, \mathcal{S}) \to \mathcal{Z}$ to map the input $x$ to a more malleable representation $z$.
    \item An Adversary function $h:\mathcal{Z} \to \mathcal{S} $ to encourage the representation in the latent space to \emph{not} be predictive of $s$.
    \item An ensemble of $\mathcal{S}$-specific decoders.
    The task is to produce a reconstruction $x_s$ from $z$ and is defined as a function $k: (\mathcal{Z}) \to \mathcal{X}_s \quad \forall ~ s \in \mathcal{S}$.
    Where $\mathcal{X}_s$ is an array of reconstructions, each corresponding to a possible $s$-value.
    During training, $\mathcal{X}_s$ is indexed by the real $s$ value so that only the $\mathcal{S}$-head that corresponds to the true protected characteristic is used for training.
\end{enumerate}


The encoder's purpose is to produce a \emph{likely} counterfactual $X$ with respect to $s$.
To do this, we produce a latent embedding, $z$ which removes as much information about $s$ as possible.
Then, we have one decoder per possible $s$-label, allowing the effect of $s$ to be reintroduced.
We train this model by optimising the objective function in Equation~\eqref{eq:ae_loss}, where $\ell_{\textrm{recon}}$ is an appropriate reconstruction loss between the reconstructions and the features, and $\ell_{\textrm{adv}}$ is the adversarial loss realised as cross-entropy between the $s$-prediction and the $s$-target coupled with a supplementary non-parametric measure (Maximum Mean Discrepancy {cite}`GreHorRasSchSmo07`) with a linear kernel.
A hyper-parameter $\lambda$ is incorporated to allow for a trade-off between the two competing losses[^9].
```latex
\begin{align} \nonumber \label{eq:ae_loss}
    \mathcal{L}_{\mathrm{AE}} = & \min_{\theta, \pi}\max_{\phi} \mathbb{E}_{x \sim X} [ \ell_{\textrm{recon}}(k_{\pi}(g_{\theta}(x), s)_s; x) \nonumber \\
    & - \lambda \ell_{\textrm{adv}}(h_{\phi}(g_{\theta}(x)), s)]
\end{align}
```

The classification model consists of a shared network with, in a similar fashion to the autoencoder, $s$-specific task-heads.
For the classification model the task is to produce an ensemble of predictions of the class label $y_s$ from $x$ and is defined as $f_s: (\mathcal{X}) \rightarrow \mathcal{Y}_s \quad \forall ~ s \in \mathcal{S}$.
As with the autoencoder, only the $\mathcal{S}$-head that corresponds to the true sensitive label is used for training.
The objective is shown in Eq `\eqref{eq:clf_loss}`.

```latex

\begin{equation} \label{eq:clf_loss}
    \mathcal{L}_{\mathrm{Clf}} = \min_{\omega, \xi} \mathbb{E}_{x \sim X}[ \ell_{\textrm{pred}}(f_{\omega}(x)_s; y) ]
\end{equation}
```

At inference time, the autoencoder model results in one reconstruction per $S$-label, per sample, and likewise for the classification model.
In the case of a binary $S$\label, this results in two reconstructions per sample and two decisions per reconstruction, resulting in 4 outcomes per sample.


## Limitations and Intended use

When we are considering an ADM deployed in a real-world biased setting, a distinction should be made between _three categories of bias_:
bias that is introduced directly by incorporating AI, due, for example, to over-fitting on the training data;
bias we can successfully intervene on by mitigating, or even completely removing, both direct and selection bias from our data and algorithm; and, bias we can detect, but cannot intervene on.

In this work, it is assumed to be possible, reasonable, and required to enforce the mapping between the observed and the decision space to be independent of the sensitive attribute at deployment, despite the data on which the model is trained not necessarily displaying this property. i.e. we assume it is a requirement to correct for direct bias within the decision environment. We achieve this by applying selection rules $2$ and $8$ from Table~\ref{table:1}. $G3$ and $G4$ within the framework allow for auditing and mitigation in the form of positive action, of bias we can detect, but cannot directly intervene on.

It is important to note the subgroup of the intersection of $G_{4}$ and $b_{3}$: when $y_{s_x=1,s_y=0}=0$ and $y_{s_x=1,s_y=1}=1$, there is a benefit to the candidate from the direct bias.
In this work, we choose to adopt a no-detriment, or positive-corrective approach.
This means that no individual, even if they are a beneficiary of past biased behaviour, will be made worse off by the positive action approach.
However, in practice, selection rules can be adapted to suit the context and objectives at hand.

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{assets/figure6.pdf}
    \caption{Changes in the engineered synthetic data, at every step, representing a simplified version of the changes different biases can have to the talent distribution of a population. Starting from a uniform distribution, we visualise how the additive effect of bias can result in a significant disproportion of success between groups differing by a sensitive attribute. In the synthetic data, lifetime and measurement bias are modelled as a shift between the distributions, while the the label bias is modelled by having a slightly different acceptance score, depending on the value of the sensitive attribute.}%
    \label{fig:SynGen2}
\end{figure*}
```
## Demonstration
To demonstrate how our approach can be applied to a candidate-filtering task within a biased setting, we use synthetic data.
This allows us to analyse the efficacy of our approach, as we can access the underlying data generation procedure to compare against ``perfect'' counterfactuals.

For this demonstration, we consider applicants to a fictitious university course.
The protected characteristic is defined as _colour_, with candidates classed as either _blue_ or _green_.

The university course is for a traditionally _blue_ profession, rendering the setting potentially biased, and the department receives applications from many more promising blue candidates compared to applications from promising green candidates.
Under these settings, we will now demonstrate the potential of our approach for bias auditing and mitigation.

### Synthetic Data Generation
We define a data generation procedure for a simple dataset with binary $S$-labels and a binary outcome, with 2 imperfect observers of 3 features, making a feature-space $\mathcal{X}$ comprising 6 features.
Full implementation details will be provided in the form of software code.
We first draw samples for $S$ from a Bernoulli distribution and model the underlying construct as a Uniform distribution - this is where the WAE axiom is applied, as $\tilde{x}_{b}$ is independent of $s$:

```latex
\begin{align*}
    s \sim \mathcal{B}(0.5) \quad\text{and}\quad
    \tilde{x}_b \sim \mathcal{U}(0,1)~~.
\end{align*}
```

(This is visualised in Figure `\ref{fig:SynGen2}` -- left.)

At this point, $\tilde{x}_b$ represents 'aptitude' for the university programme.
To represent $b_{1}$, the 'lifetime bias', for example, due to a lack of green role models at the university, or variation in parental support between blue and green parents, we map from the uniform distribution to an $S$-conditioned distribution for each feature using an inverse-CDF (percent point) function, $\tilde{x}_{b}$ to $\tilde{x}$.
This mapping is captured by $\Delta\tilde{x}_{s=0,1}$ and is shown in Figure `\ref{fig:SynGen2}` - left middle.

The features $\tilde{x}_{s=0,1}$ are still in the construct space, representing potentially useful university skills, at the point of applying.
Two noisy observations are made of these underlying skills in the form of an interview or application procedure, mapping from the construct to the observed space.
The bias potentially introduced in this measurement-process, which forms the mapping between $\tilde{x}$ and $x$, is characterised as $b_{2}$ and is demonstrated in Figure~\ref{fig:SynGen2} -- right middle.

We then generate two outcome scores.
1. An 'acceptance score' based on a linear combination of the observed features. When mapping from $x$ to $y$, both in the observed space, we add a final bias, $b_{3}$, representing direct discrimination, by setting different acceptance thresholds depending on the value of $s$.
This is demonstrated in Figure `\ref{fig:SynGen2}` -- right.
2. A 'graduation grade' based on a linear combination of the features in $\tilde{x}$, bypassing the effect of $b_2$ and $b_3$.

### Evaluation
To evaluate our model in context, we train the following models on the synthetic data:
an Oracle, \texttt{Data}, to show the disparity in the underlying data,
a Demographic Parity Oracle, \texttt{DemPar}, to put a bound on the performance of a fair decision model,
an unconstrained \texttt{Logistic Regression} model,
and established fair classification models
`K&C Reweighting` {cite}`KamCal12`,
`Kamishima` {cite}`KamAkaAso2012` and
`Fairlearn` {cite}`AgaBeyDudLanetal18` alongside
our positive action approach using `LCF` (Learnt Counterfactual) modelling.

We utilise the following metrics with results comparing the models found in Table `\ref{table:results}`[^7].

```latex
\begin{itemize}
    \item \emph{Acceptance percentage per colour}. When this is equalised across groups, demographic parity is satisfied.
        \begin{equation}
            P(Y=1|S=s) \quad \forall s \in \mathcal{S}
        \end{equation}
    \item \emph{True Capture percentage}
        \begin{align}
            P(Y=1|S&=s, G=1) \\ \nonumber
            &\cup \\
            P(Y=2|S=s, &G=1) \quad \forall s \in \mathcal{S} \nonumber
        \end{align}
    \item \emph{False Identification Difference} (FIDiff)
        \begin{equation}
            |P(G=0|s=1, Y=1) - P(G=0|s=0, Y=1)|
        \end{equation}
    \item \emph{Accuracy}(Y). The utility of the model at predicting a proxy-label based on the best assumptions from the data.
        \begin{equation}
            P(\text{prediction}=y) \quad \forall y \in \mathcal{Y}
        \end{equation}
    \item \emph{Accuracy}(G). The utility of the model at predicting the obscured `true' outcome.
        \begin{equation}
            P(\text{prediction}=g) \quad \forall g \in \mathcal{G}
        \end{equation}
\end{itemize}
```

```latex
\begin{table*}[ht]
\centering
% \resizebox{\textwidth}{!}{
\begin{tabular}{ @{}lrrcrcrr@{} }
 \toprule
 \multicolumn{1}{c}{} &
 \multicolumn{2}{c}{Theoretical Values} &
 \multicolumn{1}{c}{} &
 \multicolumn{1}{c}{Unconstrained models} &
 \multicolumn{1}{c}{} &
 \multicolumn{2}{c}{Fair models} \\
 \cmidrule{2-3} \cmidrule{5-5} \cmidrule{7-8}
  Metric & \texttt{Data} & \texttt{DemPar} && \texttt{Logistic Regression} && \texttt{Fairlearn} & \texttt{LCF} (ours) \\
 \midrule
 Acceptance|B & $35.73 \pm 0.77$ & $23.12 \pm 16.31$ && $34.54 \pm 0.79$ && $26.65 \pm 1.17$ & $\bf{35.15} \pm \bf{1.36}$ \\
 Acceptance|G & $4.30 \pm 0.33$ & $23.13 \pm 16.13$ && $5.43 \pm 0.52$ && $\bf{7.83} \pm \bf{0.52}$ & $6.04 \pm 0.60$  \\
 TCP|B $\uparrow$ & $93.82 \pm 1.36$ & $60.57 \pm 42.80$ && $91.70 \pm 1.92$ && $71.51 \pm 3.72$ & $\bf{92.63} \pm \bf{2.08}$ \\
 TCP|G $\uparrow$ & $60.38 \pm 4.73$ & $69.83 \pm 6.03$ &&  $69.84 \pm 5.00$ && $53.17 \pm 6.02$ & $\bf{96.74} \pm \bf{1.93}$ \\
 FIDiff $\downarrow$  & $0.41 \pm 0.22$ & $5.55 \pm 6.58$ &&  $0.97 \pm 0.44$ && $4.42 \pm 0.82$ & $\bf{0.86} \pm \bf{0.43}$ \\
 Accuracy(Y) $\uparrow$ & $100 \pm 0.00 $ & $84.50\pm 0.43$ && $92.64 \pm 0.41$ && $\bf{98.51} \pm \bf{0.18} $ & $98.31 \pm 0.26$  \\
 Accuracy(G) $\uparrow$  & $87.05 \pm 0.44$ & $79.25 \pm 9.48$ &&  $\bf{87.05} \pm \bf{0.46}$ && $86.16 \pm 0.44$ & $86.67 \pm 0.47$ \\
\bottomrule
\caption{Comparison table for the synthetic data results.
Theoretical values show \texttt{Data} which encode that the chance to graduate if accepted is almost independent of colour (approximately WYSIWYG since FIDiff is not $0$), and the desired values for the WAE axiom (\texttt{DemPar}).

The best result (excluding theoretical values) is highlighted in \textbf{boldface}.
Our counterfactual model (\texttt{LCF}) captures 97\% (see TCP|G row) of the candidates capable of graduating, they are either accepted or being picked up as a Positive Action Candidate (PAC).
This high TCP value is achieved while maintaining low FIDiff (as per WYSIWYG axiom).
\label{table:results}
\end{table*}
\end{tabular}
```

#### Analysing the baseline synthetic data.
As shown in Table~\ref{table:results}, the baseline data was engineered to have a high level of bias against {green} candidates:
Only $4\%$ of the {green} candidates are admitted, in comparison to the $36\%$ of the {blue} candidates.
As, unlike in the real world, we know the ground truth, we can also evaluate the True Capture percentage (TCP): how many candidates with the ability to graduate, are not being rejected. While for blue the TCP is at a high $94\%$, the green TCP is only $60\%$. The False Identification difference (FIDiff) measures how well the data conforms to `Equality of Opportunity'. A low FIDiff, $0.4\%$, means the data conforms to EqOP: once a candidate is accepted, the likelihood of graduation is nearly the same for both groups.

#### Demographic Parity Oracle.
When enforcing `perfect' demographic parity on the learning model, the metrics change substantially.
Acceptance percentage is now equal between the {blue} and {green} candidates, but at a cost: the TCP for the blue has gone down to $61\%$ and the FIDiff is up to $5.5\%$, meaning there is a higher chance for an accepted green candidate to fail, than a corresponding blue candidate.

#### Logistic Regression model.
\texttt{Logistic Regression} gives similar results to the baseline data and is used as a baseline for the comparison with subsequent models.

#### K \& C, Kamishima and Fairlearn.
None of the models manage to show significant improvement, in terms of fairness, on the baseline data.
`Fairlearn`, presented in Table~\ref{table:results} achieves the best green acceptance rate after the DP oracle, however, similar to the DP oracle, more green candidates get accepted, but not the ones capable of graduating.

#### Learnt Counterfactual Model.
Our `LCF` model accepts $40\%$ more green candidates than the baseline. These additional candidates are mostly candidates that were flagged by our model as falling victim to direct bias, and got reassigned to `accept' by the group classifier. Unlike the DP oracle (\texttt{DemPar}) and \texttt{Fairlearn}, the FIDiff remains under $1\%$, meaning the model aligns well with EqOp. A striking success, in comparison to the other models, is the high True Capture Percentage. $97\%$ of the candidates capable of graduating are not rejected, they are either accepted or being picked up as a Positive Action Candidate (PAC).

Table \ref{table:Subgroups} shows the breakdown of the outcome groups, in respect to all the candidates, produced by our `LCF` model. 
This is compared to the outcome groups of a "perfect" counterfactual (`PCF`). 
The `PCF` is the values we expect if both the encoder and classifier performed without error[^8]. 
We can see that the majority of the candidates receive a counterfactual consensus in both `PCF` and `LCF`. 
`LCF` underestimates the consensus by an overall $7\%$ when compared to the `PCF`, having larger $G_3$ and $G_4$ at the expense of the consensus subgroups. 
In this example, this means the `PCF` assigns rejection for $3\%$ more candidates then the `LCF`.


```latex
\begin{table}[h]
\centering
\begin{tabular}{ @{}ccrr@{} }
 \toprule
  Subgroup &  Outcome & \multicolumn{1}{c}{\texttt{PCF}} & \multicolumn{1}{c}{\texttt{LCF}} \\ [0.0ex]
\midrule
$g_{1}$ & 1 &$4.73 \pm 0.15$ & $3.01 \pm 0.28$ \\
$g_{2}$ & 1 &$4.73 \pm 0.09$ & $2.40 \pm 0.48$  \\
$g_{3}$ & 2 & $10.99 \pm 0.19$ & $13.77 \pm 2.08$\\
$g_{4}$ & 1 & $10.96 \pm 0.25$ & $15.27 \pm 0.91 $ \\
$g_{5}$ & 0 & $34.35 \pm 0.26$ & $32.97 \pm 2.86$\\
$g_{6}$ & 0 & $34.23 \pm 0.28$ & $32.59 \pm 0.86$ \\
\bottomrule
\end{tabular}
\caption{Comparison of the number of samples allocated to each group for a ground truth Perfect Counterfactual (\texttt{PCF}) in comparison to the Learned Counterfactuals (\texttt{LCF}).
We are able to make this comparison only because we know the ground truth for the synthetic data.
Our learned model (`LCF`) is, in general, in agreement with the ground truth (`PCF`).
}
\label{table:Subgroups}
\end{table}
```


## Experiments

### Data
In addition to the synthetic data, we evaluate our approach on the UCI Adult Income Dataset. 
This dataset comprises $45,222$ samples from the 1994 U.S. census with $14$ features including occupation, maximum attained education level and relationship status. 
Of these $14$, we reserve the binary `salary` feature as the target label, with `>$50K` as the positive outcome. 
We consider $3$ binary features as sensitive attributes: gender (Male / Female), race (White / Not White) and marital status (Married / Not Married).


```latex
\begin{figure*}[ht]
    \centering
    \includegraphics[width=\textwidth]{assets/adult_figure.pdf}
    \caption{Breakdown of group allocations on the withheld test set of the UCI Adult dataset averaged over 10 repeats, using 3 values as the protected characteristic. \textbf{Left}: The binary ``Gender'' feature. \textbf{Middle}: The ``Race'' feature binarised to membership of the majority group (white). \textbf{Right}: The ``Marital Status'' feature binarised to whether currently married. In all cases, the x-axis represents the percentage of the data that belongs to each protected characteristic, while the y-axis represents the percentage of the population assigned each outcome. Group membership is defined in Table~\ref{table:1}. For all attributes, subgroups $G3$ and $G4$ highlight the proportion of the population for which intervening on the sensitive attribute will result in the outcome changing as well. In $G3$ the outcome changes from negative to positive when $s$ changes, while changes in $G4$ result in the opposite outcome. Although an intriguing visualisation of the effect of the different attributes, conclusions should be drawn carefully as the attributes can act as a proxy to hidden patterns in the data. Further discussion can be found in Section~\ref{sec:UCI_results}.}%
    \label{fig:adult_results}
\end{figure*}
```

### Auditing the UCI Adult Income for bias
Figure `\ref{fig:adult_results}` shows a counterfactual subgroup analysis for $3$ sensitive attributes within the UCI Adult Income data set. 
We note that the accuracy of the LFC model is on par with baseline models. 
As before, the individuals within subgroups $g_{4}$ and $g_{3}$ did not achieve counterfactual consensus. For example, for gender, the subgroup $g_{4}$ contains samples of males that are above the $\$50,000$ threshold, but their female counterfactual counterparts would be under the threshold, whereas $g_{3}$ represent females under the threshold whose male counterfactual counterparts would be above the threshold.
When visually comparing the subgroups, it is possible to see that marital status is the attribute most likely to changed the outcome if flipped from married to not married an vice versa. 
This results however, should be taken with a grain of salt as marital status can also be a proxy to other attributes of influence, for example, age. 
When comparing the effects of gender and race within this data, gender seems more influential on the outcome than race. 
However, gender may also be a proxy to occupations.

Interesting patterns emerge from investigating the breakdown of candidates funnelling into $G1$ from different selection rules. 
Table~\ref{table:g1} breaks down subgroup $G_1$ further by the funnelling selection rules from Table~\ref{table:1}: selection rule 1 (Consensus), selection rules 2 and 8 (Direct bias), and other (Fallback). 
In the case of other, this indicates that bias was detected in the opposite direction to what we expect, and the decision reverted to the original outcome. 
We compare the $3$ sensitive attributes: gender, race and martial status. 
For gender, consensus is the smallest contributor, followed by correction for direct bias. 
Interestingly, the majority of contributions to $G_1$ come from 'fallback', meaning, females earning above the $50,000$ threshold, but their male counterparts would be under the threshold. 
We assume this 'reversed' bias could be present in a subset of occupations. 
Overall however, the direction of bias still favours male as $G_4$ is much larger than this subsection of $G1$. 
Race and Marital status both have Consensus as the main contributor, with fallback as a secondary contributor. 
For race and marital status, the `LCF` model didn't detect direct bias within this dataset[^10]. 
For gender, the `LCF` model detected and corrected for direct bias for $18.4\%$ of the females in subgroup $G1$.

An in-depth analysis of the UCI Adult Income data set is outside the scope of this paper, but we would like to mention a couple of easy-to-implement lines of investigations. 
First, while intervening on a protected characteristic, we allow the encoder to change all other attributes. 
It may be worthwhile to experiment with fixing other protected characteristics to eliminate proxy effects. 
Second, while examining protected characteristics separately, it is natural to assume biases may have an additive nature; this can be examined by running the LCF` model on a few sensitive attributes simultaneously. 
Extending the framework to allow multiple sensitive attribute requires expanding the selection rules. 
While lengthy, this is otherwise straightforward to derive and implement.

```latex
\begin{table}[h]
\centering
% \resizebox{\columnwidth}{!}{
\begin{tabular}{ @{}rrrr@{} }
 \toprule
  Selection Rule &  Gender & Race & Marital Status \\ [0.0ex]
\midrule
Consensus & $4.3 $ & $69.2$ & $ 71.7 $ \\
Direct bias & $18.4$ & $0 $ &  $0$ \\
Fallback & $ 77.2 $ & $30.8$ & $28.3$\\
\bottomrule
\end{tabular}%
% }
\caption{Breakdown of the $G1$ group, comprising individuals funnelled into this group due to different selection rules. Consensus corresponds to selection rule 1 in Table~\ref{table:1}. Direct bias corresponds to selection rules 2 and 8 in Table~\ref{table:1}. Fallback indicates bias was detected in the opposite direction and the decision reverted to the original outcome.}
\label{table:g1}
\end{table}
```


## Discussion and Conclusion
In this paper we present an algorithmic framework that can assist organisations in setting and following long-term diversity and equality goals, in addition to balancing a fair and practical short-term approach.

In order to facilitate both short and long term objectives, we must adopt a deferral model {cite}`MadCrePitZem18` - a model that allows for an additional outcome outside accept and reject, in our case, the `positive action candidate' outcome. This third outcome is aimed to capture and understand the needs of candidates that have high potential, but cannot yet successfully compete with capable candidates from the majority group. It also represents an idea that we have a social responsibility to try and work against disparate behaviours, including those we can not immediately correct.

The concept of 'positive action' can already be found in practice in many settings, ranging from out-reach activities, to targeted training and even adaptive policies, for example, a change in grant requirements for mothers[^11]. 
While the right 'action' will vary, and should be determined by experts, we believe that we can use AI to promote this positive practice by auditing for bias in past decisions and highlighting where missed opportunities may lie.

The 'lifetime' bias we introduce to bring together the WAE and WYSIWYG axioms is an abstract idea, representing the cumulative effect that belonging to a disadvantaged group can have on an individual's opportunities. 
We leave the interpretation as open as possible, and use it to as a blanket for all the latent contributors that lead to the statistical disparity we measure within the observed space.

The counterfactual implementation presented in this paper achieves the goal we set for it: 
it maintains predictive utility while minimising the rejection of candidates with high potential from the disadvantaged group. 
An introductory investigation of the UCI Adult Income dataset revealed several interesting patterns, including bilateral bias when gender is taken as the sensitive attribute.

We hope this work will form  part of a larger, positive discussion, around the role of AI in promoting fairness, diversity and equality.


[^1]: Originally named conditional (non-)discrimination by {cite}`KamZliCal13`.
[^2]: {cite}`YeoTsc21` makes the link between equalised odds and WYSIWYG. Equalised odds requires the True Negative Rate to be equalised in addition to the True Positive Rate. However, the link the WYSIWYG worldview can be made with both on similar grounds.
[^3]: {cite}`YeoTsc21` also presents a hybrid worldview axiom, but the approaches diverge past that point.
[^4]: As defined by the Equality Act 2010 (UK).
[^5]: We make no claims regarding the strength of 'nature' vs. 'nurture' and the framework holds for essentially all potential ratios, including either $\alpha=0$ or $\beta=0$ (but not both)
[^6]: We are notably excluding tasks that may correlate with physical attributes, for example, playing professional basketball and height.
[^7]: For ease of comparison, we choose to omit the results for `K&C Reweighting` and `Kamishima` from the table and only report `Fairlearn`, the model with the highest acceptance percentage for green candidates.
[^8]: These values assume a consistent decision rule across all populations.
[^9]: In our experiments, this is set to $1.0$.
[^10]: This is not enough for us to conclude there is no direct bias; however there is no strong enough direct pattern to be picked up by the `LCF` model within the setup of the experiment.
[^11]: For European Research Council grants, mothers get an additional 1.5 year per child added to the eligibility calculation.

## Bibliography
```{bibliography}
:filter: docname in docnames
```
