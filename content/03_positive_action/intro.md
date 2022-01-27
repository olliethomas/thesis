(ch:paper1)=
# Chapter 4: An Algorithmic Framework for Positive Action

In previous work we have identified characteristics that are overly relied on and lead to unfair decisions.
This work amends this problem statement to instead of identifying characteristics and mitigating them, identify idividuals who are at risk of receivingan unfair decision.

## Abstract
Positive action is defined within anti-discrimination legislation as voluntary, legal actions taken to address imbalance of opportunity affecting individuals belonging to under-represented groups.
We propose a novel algorithmic fairness framework that builds on this notion as a way of advancing equal representation while respecting anti-discrimination legislation and equal-treatment rights. 
We seek to identify individuals who have been negatively impacted by bias prior to application, and hence fail under an equal-treatment selection process. 
We use counterfactual fairness to assign candidates to one of three groups: 
1) Candidates who would have been successful with any set of perceived protected attributes are assigned a successful outcome.
2) Candidates who would have remained unsuccessful with any set of protected attributes are assigned an unsuccessful outcome.
3) Unsuccessful candidates that would have been successful if they were born with a different set of protected attributes are flagged as positive action candidates.

(sec:intro)=
## Introduction

Allocating resources amongst individuals, for example, jobs or university placements, requires us to evaluate an individual’s suitability for the task. 
We also want to ensure the selection process is fair and that positive outcomes are fairly distributed within the population. 
Machine Learning (ML) systems are increasingly being used to inform, support, or even directly make decisions within these consequential domains, affecting millions of lives {cite}`BarSel16`. 
It is therefore necessary to consider how the notions of fair process and fair outcomes translate into algorithmic decision support frameworks {cite}`Wachter20, Xiang21`.

Anti-discrimination legislation in the E.U., U.S. and U.K. among others, indicate that a fair selection process requires equal treatment in the sense that protected attributes, for example, gender and race, should not be considered within the decision making process without a good reason {cite}`DwoHarPitReiZem12, Ben19`. 
Ignoring the protected attributes within an algorithmic approach, however, guarantees neither fair process nor fair outcome {cite}`PedRugTur08, HarWal19, Wachter20, Xiang21, SimBhaWel21`. 

Decision support algorithms are commonly trained on a dataset of past decisions. 
The resulting model may disproportionately predict positive outcomes in favour of the majority over historically under-represented groups {cite}`KamCal12,KalZho18`. 
These statistical disparities could arise from two distinct mechanisms: 
(i) unequal treatment;  
or (ii) equal treatment when the status quo itself is not neutral. 
The former occurs when the data contains discriminatory past decisions. 
The latter, when historically under-represented groups struggle to compete with the majority under an _equal-treatment selection process_ – a selection process that is `blind' to the applicant's protected attribute. 
Often, the statistical disparity in the training data, and as a result, in the model’s prediction is a combination of both.

Enforcing _Demographic Parity_ (DP) – an equal fraction of positive outcomes across subgroups – is usually impractical. 
In addition to hindering the accuracy of the model’s predictions, this approach will not typically align with anti-discrimination legislation. 
A common alternative is to impose an algorithmic fairness constraint that better aligns with the notion of equal treatment, and maintain a disparity in the positive outcome rates {cite}`HarPriSre16, Wachter20`. 

Anti-discrimination legislation acknowledges the need to bridge the gap between equal treatment and equal representation. 
The Equality Act 2010 (UK) defines _positive action_ as "lawful measures taken to encourage and train people from under-represented groups to help them overcome disadvantages in competing with other applicants". 
````{margin}
```{note}
European legislation defines positive action similarly. In the US similar measures can be employed under affirmative action, however, the definition does not completely overlap with positive action.
```
````  
Examples of positive action include, but are not limited to: 
additional training opportunities and mentoring programs available to the under-represented group;
targeted advertising, outreach, networking and bursaries. 
Policies designed to meet the needs of under-represented groups are also considered positive action. 
For example, the European Research Council introduced extensions of eligibility for women with children, in addition to modified CV templates for applicants and anti-bias briefings for the members of evaluation panels.
````{margin}
```{note}
These measures are included in the European Research Council's Gender Equality Plan for 2021-2027.
```
```` 
The action taken is required to be 'proportionate' to both the extent and longevity of the under-representation, and to the barriers the under-represented group experience. 

We argue that incorporating the notion of positive action within decision support algorithms can advance the use of these measures, promoting equal representation while respecting anti-discrimination legislation and equal-treatment rights. 
In this work we propose a novel algorithmic fairness framework to identify _positive action candidates_ — individuals who were negatively impacted by earlier bias, and hence fail under an equal treatment process. We compare our approach to a baseline of choosing the top rejected candidates from the under-represented group in section {ref}`sec:compare` ({numref}`fig:compare`).

```{figure} ./assets/figure1.png
---
width: 100%
name: fig:story
---
A 'hybrid' worldview showing biases potentially introduced at each step.
The aptitude is assumed to be independent of all protected attributes, aligning with the WAE worldview.  
By the point of observation, however, the construct space might have altered and now  disparity between sub-groups is allowed, aligning with the WYSIWYG worldview. Opportunity bias, selection bias, measurement bias and label bias can introduce or further aggravate the disparity between the protected group and the majority.
```

(sec:background)=
## Background
(sec:background_definitions)=
### Definitions
In this work we discuss sub-groups with respect to _protected attributes_ – characteristics that, by law, must not be the basis for discrimination. 
These include, but are not limited to, race, gender, age, religion and disability. 
We define a protected subgroup as an under-represented group separated from the majority by a protected attribute.
For example, women in the engineering profession are under-represented when compared to their representation within the population. 
In the context of a decision support system, we may observe a statistical disparity — a disproportionate positive outcome (hiring, admissions) rate — in favour of the majority, compared to the protected subgroup. 
This can be as a result of the model being trained on past discriminatory decisions, but can also be the result of a statistical disparity in the input features (grades, qualifications). 
In this work, we define bias as a mechanism by which statistical disparity between a protected subgroup and the majority emerges or is exacerbated. 
Bias within the decision making process will affect the decision outcome. 
Bias that occurred prior may affect the features. 
To expand the discussion on bias, it is useful to briefly refer to the framework presented in {cite:p}`FriSchVen16`, which defines three spaces – the _construct space_, _observed space_ and _decision space_ – and uses the mappings between them to formalise several definitions of bias.

The _construct space_ represents the _'ground truth'_ – an unobserved space that correctly captures differences between individuals with respect to a task; 
the _observed space_ represents the measurable features for consideration, and the _decision space_ the outcome {cite}`FriSchVen16`. 
For example, intelligence resides in the construct space, measured IQ resides in the observed space, and acceptance or rejection from the International Mensa club resides in the decision space. 

As the observed space is an estimate of the construct space, we can make certain assumptions regarding the mapping between these spaces. 
These assumptions can be referred to as _worldviews_. 
Prior work {cite}`FriSchVen16` established the worldviews WAE ("we're all equal") and WYSIWYG ("what you see is what you get"), that are considered to be in tension with each other. 
WAE assumes that we are all the same in construct space. 
If that is not the case in the observed space, this disparity is attributed to structural bias — an incorrect mapping between the construct and the observed space.
WYSIWYG, on the other hand, allows for a disparity between protected subgroups, assuming the observed discrepancies are a true reflection of disparities in the construct space. 

To better understand the statistical disparity we may observe in the data, we discuss specific types of bias. 
_Sample selection bias_ originates from training on a non-representative sample of the population {cite}`Tol19`.
_Label bias_ occurs when the dataset contains past discriminatory decisions {cite}`WicPanTri19,JiaNac20`. 
Mitigation efforts that consider selection bias {cite}`KamCal12,AgaBeyDudLanetal18`, or label bias {cite}`CalVer10,JiaNac20,KehCheQua20` independently are available. 
Bias can also be introduced outside of the environment which we can control, i.e. outside the training population, measurements and learning algorithm. 
The identification of positive action candidates within decision support systems using the framework presented in this paper aims to acknowledge and mitigate a broader range of biases. 
This includes bias that cannot normally be mitigated by an automated rejection / acceptance model while respecting anti-discrimination legislation and the right for equal-treatment.

### Counterfactual Modelling
To identify positive action candidates, we take a counterfactual approach. 
A counterfactual outcome is a hypothetical outcome for a scenario that is identical in all respects except for a specific, well-defined change and its causal consequences {cite}`Hume00, Miller19`. 
In the context of this work, we focus on counterfactual scenarios with respect to a change in a protected attribute, and distinguish between two types of counterfactual questions:
1. Would the outcome change if \textbf{only} the protected attribute was different?
2. Would the outcome change if the protected attributed and its \textbf{causal consequences} were different?

For example, if a female applicant is not invited for a job interview, we can ask the following two questions: if her CV was identical, but the application *appeared* to be from a male applicant, would she be invited to interview? {cite}`BerMul04`
If she had been *born* male, experienced life as a male, and then applied for the same job, would she have been invited for an interview?

The second counterfactual question is critical to our approach, as it is used to identify positive action candidates. 
We use the first counterfactual question to detect and mitigate label bias. 

To evaluate counterfactual outcomes, ideally, we would rely on a Structural Causal Model (SCM) — a graphical model whose vertices represent features and whose edges  represent the _causal_ pathway between them {cite}`Pearl09`. 
The first type of counterfactual question, for example, will manifest in a SCM as a direct pathway between the protected attribute and the outcome.
This SCM-based approach is taken in works related to fairness such as {cite}`KusLofRusSil17,Chi19`.
However, a complete structural model, is challenging to obtain — they are application specific and require specification by domain experts. 
In practice, we can find two as-close-as-possible individuals (differing by the protected attribute) within the data (e.g. {cite}`RosRub83,Rubin90`) or by creating the counterfactual representations using an adversarial learning model (e.g. {cite}`MadCrePitZem18b,ShaHenDarQua20,GoeGuLiRe20`).
In this paper, we employ the latter approach.

```{figure} ./assets/compare_figure.png
---
height: 300px
name: fig:compare
---
How our approach to choosing positive action candidates compares to choosing the top rejected candidates from the under-represented group. 
With an equal weight selection role, Applicants A and B have the same overall score.
Re-scaling the minority's maths grade distribution to match the majority's distribution highlights applicant B as the better positive action candidate.
```

### Related Works
As far as we are aware, this is the first work to address positive action within the context of a decision support system.
However, there are prior works that look at related problems.
We briefly describe the most relevant ones to place the problem of determining positive action candidates in context.

#### Deferral
The challenge in learning to defer is identifying which candidates the model is uncertain about. 
Once identified, these candidates are referred to a human decision-maker which comes at some cost {cite}`MozSon20,MadPitZem18`.
This poses interesting questions about the practical quantification of uncertainty, and would be a potential extension to our framework.
However, deferment differs from identifying  positive action candidates, as the system we are adapting may be confident in it's assessment that a candidate who would be suitable for positive action should be rejected.

#### Actionable Recourse
Another related field is that of recourse.
Works in this area, such as {cite}`UstSpaLiu19,JosKoyVijKimGho19,KarSchVal21` aim to determine how the world would have had to be different for an alternative outcome to occur.
They aim to explain what would need to change about a rejected candidate for them to be accepted.
Our framework instead asks more direct questions: 
If a candidate were perceived to have an alternative protected attribute value, would the outcome be different? 
And, would the outcome change if the protected attribute and its causal consequences were different?

#### Auditing Systems
This is a multi-faceted, broad area, but in general auditing aims to evaluate either a dataset {cite}`SalKueSteAniHinLonGha18`, or a system {cite}`KeaNeeRotWe18` for potential bias. 
Examples of auditing systems that are similar to ours include {cite}`BlaYeoFre20`. 
In their work the authors take an alternative counterfactual approach based on finding the nearest datapoint in the data with a different protected attribute and compare outcomes. 
These works differ in motivation as they use their auditing method to look at which _groups_ are most affected, whereas we evaluate which _individuals_ are likely to be affected.

(sec:approach)=
## Approach
We propose a novel algorithmic fairness framework for advancing equal representation while respecting anti-discrimination legislation and the right for equal treatment:
1. We identify _positive action candidates_: individuals who were negatively impacted by earlier bias, and hence fail under an equal treatment process.
2. Our approach is to use counterfactual fairness to assign applicants into groups:
    1. Applicants who would have been successful with any set of perceived protected attributes are assigned a successful outcome;
    2. Applicants who would have remained unsuccessful with any set of protected attributes are assigned an unsuccessful outcome;
    3. Unsuccessful applicants that would have been successful if they were born with a different set of protected attributes are flagged as positive action candidates.

(sec:ours)=
### Positive Action Framework

#### Fairness Worldview
Where does the concept of positive action fit within technical fairness definitions?
If we consider WAE and WYSIWYG, the worldviews discussed in section {ref}`sec:background_definitions`, the notion of determining positive action candidates does not fully align with either worldview. 
The 'hybrid' worldview adopted in this work is illustrated in Figure {numref}`fig:story` and in Figure {numref}`fig:our_model` as a graphical model.
 
We expand  the _construct space_ to include the element of _time_, with the observed space representing measurements of the construct space over time. 
We assume equality was true at some stage, but is not necessarily so at the time of measurement. 
We therefore align with WYSIWYG at the point of measurement, and assume that statistical disparity observed within the observed space is, at least partly, due to disparity within the construct space. Positive action is aimed at addressing the disparities within the construct space as we  assume those occur due to imbalance of opportunity or disadvantage affecting individuals belonging to the under-represented group.

To illustrate this, consider a set of measurable features $X$ within the observed space $\mathcal{X}$, where $\mathcal{X}$ represents the space of all potential feature values. 
Each individual sample $x \in X$ is an approximation to its non-measurable construct-counterpart $\tilde{x} \in \tilde{X}$.

\begin{equation}
  X\approx\tilde{X}=\alpha\cdot\tilde{X}_{apt}+\beta\cdot\Delta\tilde{X}
\end{equation}
where 
\begin{align*}
  \tilde{X}_{apt}\perp S \quad\text{and}\quad
  \Delta\tilde{X} \not \perp S~~,
\end{align*}
where $S$ is the protected attribute, and $\alpha$ and $\beta$ are non-negative values that sum to 1.
In words, we assume an individual’s suitability for the task, at the time of measurement, is a combination of their aptitude ($\tilde{X}_{apt}$), a natural born ability, and their experiences over time ($\Delta \tilde{X}$).
````{margin}
```{note}
We make no claims regarding the strength of `nature' vs. `nurture'. The framework holds for all potential ratios, including either $\alpha=0$ or $\beta=0$.
```
````
We further assume that the aptitude component, $\tilde{X}_{apt}$, is independent of any protected attribute, and hence complies with the WAE worldview.
````{margin}
```{note}
We are excluding tasks that may correlate with physical attributes, for example, playing professional basketball and height.
```
````
The 'life-experience' component $\Delta\tilde{X}$, shifts the aptitude either positively or negatively, and may not be independent of $S$.
$\tilde{X}$ represents the non-observable 'ground truth' at the time of measurement, which could be dependent on $S$. 
A graphical model of our worldview is shown in Figure {numref}`fig:our_model.

```{figure} ./assets/figure2_rev.png
---
height: 200px
name: fig:our_model
---
The effect of a protected attribute $S$ on descendants of $\tilde{\mathcal{X}}_{apt}$ throughout a data-generation procedure. 
$\tilde{\mathcal{X}}$ within the construct space, $\mathcal{X}$ within the observed space and $\mathcal{Y}$ within the decision space.
```

#### Underlying mechanisms and bias

We consider a setting where we observe a statistical disparity between subgroups separated by the value of $\mathcal{S}$, within both the observed space and the decision space. 
The disparity within the decision space may be worse than the disparity within the observed space. 
One mechanism that can cause this aggravation is label bias – a direct impact of the protected attribute $\mathcal{S}$ on the outcome $\mathcal{Y}$ due to past discriminatory decisions within the training dataset. 
To achieve equal treatment, the effects of label bias should be eliminated. 

The disparity within the observed space can be caused by several mechanisms or their combination:
selection bias occurs when the training set contains a non-representative sample of the population;
measurement bias occurs when the mapping from the construct space to the observed space isn't as faithful for certain groups or individuals.

Part of the disparity within the observed space can be a true reflection of a disparity within the construct space itself, at the time of measurement. 
We assume that the distribution of aptitude $\mathcal{X}_{apt}$ in the contract space is the same across subgroups. 
While a variation in the opportunities available to different individuals is normal, when an imbalance of opportunities affects a protected group more than the majority, it can result in a disparity within the construct space itself. 
Addressing this imbalance of opportunities is a principal component of positive action and our framework.
````{margin}
```{note}
We note that this is not an extensive discussion of bias and there are other underlying mechanisms that can lead to a statistical disparity between an under-represented group and the majority.
```
````

### Positive Action Candidates


```{figure} ./assets/figure3.png
---
height: 300px
name: fig:gcm_redux
---
_Top row_:
    _Left_: The accepted ($y=1$) and rejected ($y=0$) ratios difference between a protected group ($s=0$) and the majority ($s=1$), under an equal-treatment selection rule (WYSIWYG worldview). 
    _Right_: The accepted and rejected ratios difference between protected group and the majority when demographic parity is enforced (WAE worldview).
_Bottom_: Overlapping the above two worldviews.
The population captured by groups $G_{1}$, $G_{2}$, $G_{5}$ and $G_{6}$ have consistent outcome across both worldviews.
Groups $G_{3}$ and $G_{4}$ represent individuals that will receive different outcome if a different worldview is assumed.
```

```{figure} ./assets/model.png
---
width: 100%
name: fig:wae_wys_thirdoption
---
Diagram illustrating our method. 
The original representation $x$ is mapped to a representation $z$ that is independent of the protected attribute $s$. 
The invariant representation $z$ is then mapped back into both $x_{s_x=0}$ and $x_{s_x=1}$, reintroducing biases associated with each subgroup. 
Each of those representations is labelled, resulting in four representation in total. 
The four corresponding predicted outcomes then determine the group classification according to one of three final outcomes: _accept_, _reject_, or _disagreement_ which has two outcomes associated. 
Candidates from under-represented groups that were rejected, but would have received a positive outcome in a counterfactual world are _flagged for positive action_. 
Candidates from over-represented groups that were flagged for acceptance, but would _not_ have received a positive outcome in a counterfactual world remain accepted under a 'no-detriment' policy.
```

#### Quantifying the difference between WAE and WYSIWYG

To quantify the difference between the WAE and the WYSIWYG worldviews we divide the data into six subgroups, as shown in Figure {numref}`fig:gcm_redux`. 
This procedure be done for any pair of fairness metrics and definitions. 
We compare positive outcome ratios between an equal-treatment selection rule and demographic parity, metrics associated with WYSIWYG and WAE, respectively. 
We conceptually overlay the observed data (Figure {numref}`fig:gcm_redux`, top left) on a representation of the data with demographic parity enforced (Figure {numref}`fig:gcm_redux`, top right). 
When overlaid, the data can be separated into six subgroups, as shown in Figure {numref}`fig:gcm_redux`, bottom. 
Subgroups $G_{1}$ and $G_{2}$ get a positive outcome under both worldviews. 
Subgroups $G_{5}$ and $G_{6}$, get a negative outcome under both worldviews. 
Subgroups $G_{3}$ and $G_{4}$, however, represent _a different outcome under the two worldviews_. 
Subgroup $G_{3}$, represents the subgroup that would have received a positive outcome had demographic parity been enforced, and a negative outcome based on the observed data. 
This subgroup may be interpreted as individuals who were negatively impacted by earlier bias, and hence fail under an equal-treatment selection process. 
We cannot accept these applicants while aligning with anti-discrimination legislation. 
We can highlight them as candidates for positive action — high-potential applicants from an under-represented group whom targeted positive action can help succeed under a equal-treatment selection process.

```{table} Selection rules for mapping from the groups represented in Figure~\ref{fig:gcm_redux} and Figure~\ref{fig:wae_wys_thirdoption} to a decision. As $s=0$ represents an disadvantaged group, we identify those in group 3 to be suitable for \emph{positive action}. These are candidates who would have been accepted had a counterfactual version of themselves been considered. N.b. Combinations not listed are identified and the outcome reverts to the outcome from an unconstrained model.
:name: table:1
| Selection Rule |   $s$  | $y_{s_x=0,s_y=0}$ | $y_{s_x=0,s_y=1}$ | {$y_{s_x=1,s_y=0}$} | $y_{s_x=1,s_y=1}$ |      Subgroup      |        Bias        | $y$ |
|----------------|:------:|:-----------------:|:-----------------:|:-------------------:|:-----------------:|:------------------:|:------------------:|:---:|
| 1              | 0 or 1 |         1         |         1         |          1          |         1         | $G_{1}$ or $G_{2}$ |          -         |  1  |
| 2              | 0 or 1 |         0         |         1         |          1          |         1         | $G_{1}$ or $G_{2}$ |       $b_{3}$      |  1  |
| 3              |    1   |         0         |         0         |          1          |         1         |       $G_{4}$      | $b_{1}$ or $b_{2}$ |  1  |
| 4              |    1   |         0         |         0         |          0          |         1         |       $G_{4}$      |       $b_{3}$      |  1  |
| 5              |    1   |         0         |         1         |          0          |         1         |       $G_{4}$      |       $b_{3}$      |  1  |
| 6              |    0   |         0         |         0         |          1          |         1         |       $G_{3}$      | $b_{1}$ or $b_{2}$ |  2  |
| 7              |    0   |         0         |         0         |          0          |         1         |       $G_{3}$      |       $b_{3}$      |  2  |
| 8              |    0   |         0         |         1         |          0          |         1         |       $G_{1}$      |       $b_{3}$      |  1  |
| 9              | 0 or 1 |         0         |         0         |          0          |         0         | $G_{5}$ or $G_{6}$ |          -         |  0  |
```

(sec:compare)=
#### Choosing the right positive action candidates

Demographic parity is a group fairness measure that compares the ratios of positive outcome rates between subgroups. 
We still need to identify which applicants we want to highlight as positive action candidates. 
Our approach is to use counterfactual fairness to identify the most suitable applicants.
We highlight unsuccessful applicants that would have been successful if they were born with a different value of $S$, the protected attribute.

The reader might now consider the merit of this approach compared to the more straightforward 'baseline' approach of highlighting the top rejected candidates from the under-represented group. 
This baseline is only applicable when there is a clear way to rank candidates and does not account for two potential issues: measurement bias and an uneven dispersion of disparity amongst the input features.
We illustrate these two issues with the following motivating example: 

Consider a minority who traditionally sends their children to schools that teach English to a good level but teaches mathematics only to a basic level. 
This minority is under-represented within STEM subjects. 
A good level of mathematical ability is crucial for succeeding in STEM university courses and a high grade in a Maths exam is a key part of the selection process. 
To keep this example simple, we consider the application to consist of grades in only two subjects, Maths and English, with equal weight. 
Blindly taking the best rejected applicants will not spot the applicants who did exceptionally well in Maths, considering the poor education they received in this subject. 
In our approach, the minority's Maths grade distribution gets re-calibrated to match the majority's distribution, while the distribution of the English grades is left unaffected because there is no disparity with the majority's distribution. 
This means that a minority applicant who is good in Maths, relative to their minority subgroup, will be preferred compared to one that is relatively good in English. 
Figure {numref}`fig:compare` illustrates how two applicants would be ranked under our approach compared the baseline of choosing the top rejected candidates. 
For the majority, the distribution ranges between 0-10 for both English and maths. 
For the minority, the English distribution ranges between 1-10 but the maths distribution only ranges between 1-5. 
Applicant A's grades are 2 and 9 in maths and English, respectively. 
Applicant B's grades are 5 and 6 in maths and English, respectively. 
With an equal weight selection rule, they both have an overall score of 11. 
When we re-scale the maths grade distributions of the minority to match the majority's distribution, applicant B is highlighted as the better positive action candidate with an overall score of 16 compared to 13 for applicant B. 
This re-calibration is only put into effect when populating the positive action candidates group.
When applicants are considered to decide if they should receive an accept outcome, the features are taken as they are. 
In the case of this example, we may not be able to accept applicant B, but they are flagged as a positive action candidate — a maths foundation course, for example, is likely to allow them to successfully compete in a subsequent selection process. 


### Implementation
#### Building a group classifier.
To identify which candidates may benefit most from positive action we use a general two-step approach following the scheme in Figure {numref}`fig:wae_wys_thirdoption`.
Our aim is to produce, with respect to a protected attribute, both counterfactual samples; and counterfactual decisions.
The first accounts for differences in the features.
The latter accounts for decisions that are potentially discriminatory (e.g. past positive-discrimination attempts).


We use an approach from fair representation literature to make a representation of the data that, as best possible, is invariant to $S$.
First, we train an adversarial autoencoder model that maps the observed data point $x$ from the dataset $X$ into a latent representation $z \in \mathcal{Z}$ (where $\mathcal{Z}=\mathbb{R}^{N_z}$), that is independent of the protected attribute, $s \in \mathcal{S}$, where $\mathcal{S}$ is the set of possible protected attribute values. For example, $\mathcal{S} = \{0, 1\}$ if the protected attribute is binary.
From an instance $z$, two mirror representations can be created, $x_{s_x=0}$ and $x_{s_x=1}$. 
The variables $x_{s_x=0}$ and $x_{s_x=1}$ are then labelled by concatenating the perceived protected attribute to the covariates, creating four representation in total: $x_{s_x=0,s_y=0}$, $x_{s_x=0,s_y=1}$, $x_{s_x=1,s_y=0}$ and $x_{s_x=1,s_y=1}$. 
Here, $S_y$ denotes the value of the protected attribute concatenated to the set of covariates, adding a direct path in the data to $S$.
A classifier can use this value directly if it is indeed the basis of a decision, rather than extracting the protected attribute from the remaining features.

For the second step, we train a second model, a shared classifier to perform predictions on the counterfactual representations. 
We then feed in the counterfactuals and get a corresponding set of outputs:
$y_{s_x=0,s_y=0}$, $y_{s_x=0,s_y=1}$, $y_{s_x=1,s_y=0}$ and $y_{s_x=1,s_x=1}$. 
With this knowledge, the Outcome Comparator then sorts the set of original datapoints $X$ into one of six subgroups $G_{1-6}$. 
The full selection rules are presented in Table {numref}`table:1`, but we give some intuition: 

Groups 1 \& 2 ($G_{1,2}$) consist of candidates whose outcomes were either unanimously accepted across all counterfactual inputs (selection rule 1), or differed due to $S_y$, the concatenated _perceived_ protected attribute, changing (selection rules 2 \& 8).
Unanimous _negative_ outcomes for all counterfactual inputs are assigned to groups $G_{5,6}$ (selection rule 9). 
Lastly, applicants who receive a disagreement amongst the outcomes, i.e. their outcome depends on the value of $S_x$, are assigned to groups $G_{3,4}$ (selection rules 3-7). 
Members of group $G_4$ are accepted as they would by an unconstrained classifier as our positive action approach has no-detriment to the over-represented group.
Members of group $G_{3}$ are highlighted as _positive action candidates_.


#### Model
Our model is implemented as two successive neural networks representing distinct phases as mentioned above.

The adversarial autoencoder has a similar architecture to {cite}`MadCrePitZem18`, with multiple decoders {cite}`ShaJohSon17,ParChuChaLeeShi21`, and comprises:
1. An encoder function $g: (\mathcal{X}, \mathcal{S}) \to \mathcal{Z}$ to map the input $x$ to a more malleable representation $z$.
2. An Adversary function $h:\mathcal{Z} \to \mathcal{S} $ to encourage the representation in the latent space to \emph{not} be predictive of $s$.
3. An ensemble of $\mathcal{S}$-specific decoders.
   The task is to produce a reconstruction $x_s$ from $z$ and is defined as a function $k: (\mathcal{Z}) \to \mathcal{X}_s \quad \forall ~ s \in \mathcal{S}$.
   Where $\mathcal{X}_s$ is an array of reconstructions, each corresponding to a possible $s$-value.
   During training, $\mathcal{X}_s$ is indexed by the real $s$ value so that only the $\mathcal{S}$-head that corresponds to the true protected attribute is used for training.

The encoder's purpose is to produce a _likely_ counterfactual $X$ with respect to $S$.
To do this, we produce a latent embedding, $Z$ which removes as much information about $S$ as possible. 
Then, we have one decoder-head per possible $S$-label, allowing the effect of $s$ to be reintroduced.
````{margin}
```{note}
This could be performed with a conditional decoder that additionally accepts the protected attribute as input, but in practice, we found our approach to work more consistently.
```
````
We train this model by optimising the objective function in Equation {ref}`eq:ae_loss`, where $\ell_{\textrm{recon}}$ is an appropriate loss between the reconstructions and the features, and $\ell_{\textrm{adv}}$ is the adversarial loss realised as cross-entropy between the predicted and target $S$ coupled with a supplementary non-parametric measure (Maximum Mean Discrepancy {cite}`GreHorRasSchSmo07`) with a linear kernel between the embeddings per group (i.e. $\textrm{MMD}(Z_{s=0}, Z_{s=1})$ ).
A hyper-parameter $\lambda$ is incorporated to allow for a trade-off between the two competing losses.
````{margin}
```{note}
In our experiments, we use $\lambda = 1.0$
```
````
\begin{align} \nonumber \label{eq:ae_loss}
    \mathcal{L}_{\mathrm{AE}} = & \min_{\theta, \pi}\max_{\phi} \mathbb{E}_{x \sim X} [ \ell_{\textrm{recon}}(k_{\pi}(g_{\theta}(x), s)_s; x) \nonumber \\
    &\qquad\qquad\qquad\quad - \lambda \ell_{\textrm{adv}}(h_{\phi}(g_{\theta}(x)), s)] 
\end{align}

The classification model consists of a shared network with, in a similar fashion to the autoencoder, $S$-specific task-heads.
This is to capture any potential direct discrimination that the model determines to exist based on past data.
For the classification model the task is to produce an ensemble of predictions of the class label $y_s$ from $x$ and is defined as $f_s: (\mathcal{X}) \rightarrow \mathcal{Y}_s \quad \forall ~ s \in \mathcal{S}$. 
As with the autoencoder, only the $\mathcal{S}$-head that corresponds to the true protected label is used for training. 
The objective is shown in the following equation:

\begin{equation} \label{eq:clf_loss}
    \mathcal{L}_{\mathrm{Clf}} = \min_{\omega, \xi} \mathbb{E}_{x \sim X}[ \ell_{\textrm{pred}}(f_{\omega}(x)_s; y) ]
\end{equation}

At inference time, the autoencoder model produces one reconstruction per $S$-label, per sample, and likewise for the classification model.
In the case of a binary $S$\label, this produces two reconstructions per sample and two decisions per reconstruction, resulting in 4 outcomes per sample.


```{figure} ./assets/figure6.png
---
height: 200px
name: fig:SynGen2
---
Changes in the engineered synthetic data. 
Starting from a uniform distribution, we visualise how the additive effect of bias can result in a significant disproportion of success between groups differing by a protected attribute. 
The opportunity bias and measurement bias are modelled as a shift between the distributions. 
The label bias is modelled by having different acceptance thresholds for the different groups (vertical dashed lines in the right figure).
```
(sec:demo)=
## Experiments
We first use synthetic data to demonstrate how our approach can be applied to a candidate-filtering task within a biased setting.
We consider applicants to a university course in a fictitious world that is inhabited by _blue_ and _green_ people, such that we take a person's _colour_ as the protected attribute.
This university course is for a traditionally _blue_ profession, rendering the setting potentially biased. 
The department receives applications from many more promising _blue_ candidates than from promising _green_ candidates.

We then demonstrate our approach on the UCI Adult Income dataset, and use it to highlight potential challenges in real-world deployment. 

### Data
(sec:synthetic_generation)=
#### Synthetic Data Generation

We define a data generation procedure for a dataset with binary $S$-labels and a binary outcome, with $2$ imperfect observers of $3$ features, making a feature-space $\mathcal{X}$ comprising 6 features. Full details are in Appendix {ref}`app:synthetic`.

We first draw samples for $S$ from a Bernoulli distribution, and model the underlying construct as a Uniform distribution (Figure {numref}`fig:SynGen2`(i)) — this is where the WAE worldview is applied, as $\tilde{X}_{apt}$ is independent of $S$:
\begin{align*}
    S \sim \mathcal{B}(0.5) \quad\text{and}\quad
    \tilde{X}_{apt} \sim \mathcal{U}(0,1)~~.
\end{align*}

To represent the imbalance of opportunity between the groups, for example, due to variation in parental support between blue and green parents, we map from the uniform distribution to an $S$-conditioned distribution for each feature using an inverse-CDF (percent point) function, $\tilde{X}_{apt}$ to $\tilde{X}$. 
This mapping is captured by $\Delta\tilde{X}_{s=0,1}$ (Figure {numref}`fig:SynGen2`(ii)).

The features $\tilde{X}_{s=0,1}$ are still in the construct space, representing the potential to  successfully graduate from the university course at the point of applying. 
The mapping between $\tilde{X}$ and $X$ is made of two noisy observations for each feature.
A measurement bias further aggravates the disparity between the blue and green distributions (Figure {numref}`fig:SynGen2`(iii)). 
We then generate two outcome scores:
1. An 'acceptance score' based on a linear combination of the observed features. 
   When mapping from $X$ to $Y$, from the observed to the decision space, we add a label bias by setting different acceptance thresholds depending on the value of $S$ (Figure {numref}`fig:SynGen2`(iv)).
2. A 'graduation grade' based on a linear combination of the features in $\tilde{X}$, bypassing the effect of the introduced measurement bias and label bias.

#### UCI Adult Income Data
We evaluate our approach on the UCI Adult Income Dataset, which is often used for evaluating fairness-enhancing systems. 
This dataset comprises $45,222$ samples from the 1994 U.S. census with $14$ features including occupation, maximum attained education level and relationship status. 
Of these $14$, we reserve the binary `salary` feature as the target label, with `>$50K` as the positive outcome. 
We consider $3$ binary features as protected attributes: sex (Male/Female), race (White/Not White) and marital status (Married/Not Married).


\subsection{Evaluation} 
To evaluate our model in context, we train the following models on the synthetic data:
a Demographic Parity Oracle, \texttt{DemPar}, enforcing exact Demographic Parity; an unconstrained \texttt{Logistic Regression} model;
established fair classification models 
\texttt{K \& C Reweighting} \cite{KamCal12}, 
\texttt{Kamishima} \cite{KamAkaAso2012} and 
\texttt{FairLearn} \cite{AgaBeyDudLanetal18}; 
and our positive action approach using counterfactual modelling, which we refer to as \texttt{PAF} (Positive Action Framework).

We define the following metrics:

_Acceptance percentage per colour_ (Acceptance). When this is equalised across groups, demographic parity is satisfied.
    \begin{equation}
        \textrm{Acceptance}(s)=P(Y=1|S=s) \quad \forall s \in \mathcal{S}
    \end{equation}
With $Y=1$ being the 'accepted' outcome.
    
_True Capture percentage_ (TCP). This captures the rate of applicants with the ability to graduate, that are not rejected: 
    \begin{align}
        \textrm{TCP}(s)=P(Y\in\{1, 2\}|S=s, G=1) \quad \forall s \in \mathcal{S} %\nonumber
    \end{align}
With $Y=2$ being the `positive action candidate' outcome.
    
_False Identification Difference_ (FID) measures the level of `Equality of Opportunity' (EqOP), i.e. once a candidate is accepted, does their chance of graduating depend on the protected attribute? It is calculated as:  
    \begin{equation}
        \textrm{FID}=|P(G=0|S=1, Y=1) - P(G=0|S=0, Y=1)|
    \end{equation}
         
_Accuracy_. We evaluate the utility of the model with regard to both $Y$, predicting a proxy-label based on the best assumptions from the data; and $G$, predicting the obscured 'true' outcome.
    \begin{align}
        & \textrm{Accuracy}(y)=P(\text{prediction}=y) \quad \forall y \in \mathcal{Y} \\
        & \textrm{Accuracy}(g)=P(\text{prediction}=g) \quad \forall g \in \mathcal{G}
    \end{align}
        
    
\begin{table*}[ht]
\caption{Comparison table for the synthetic data results. 
Oracle values show the desired values for the WAE worldview (\texttt{DemPar}). 
The best result (excluding oracle values) is highlighted in \textbf{boldface}.
Our positive action framework model (\texttt{PAF}) captures 97\% (see TCP|G row) of the green applicants capable of graduating: they are either accepted or flagged as positive action candidates.
This high TCP value is achieved while maintaining low FID (as per WYSIWYG worldview). 
}


```{table} Comparison table for the synthetic data results. Oracle values show the desired values for the WAE worldview (\texttt{DemPar}). The best result (excluding oracle values) is highlighted in \textbf{boldface}. Our positive action framework model (\texttt{PAF}) captures 97\% (see TCP|G row) of the green applicants capable of graduating: they are either accepted or flagged as positive action candidates. This high TCP value is achieved while maintaining low FID (as per WYSIWYG worldview). 
:name: table:results
|                        | Theoretical Values |   |     Unconstrained models     |   |         Fair models         |                            |
|:----------------------:|:------------------:|:-:|:----------------------------:|:-:|:---------------------------:|---------------------------:|
| Metric                 |    \texttt{DemPar} |   | \texttt{Logistic Regression} |   |      \texttt{Fairlearn}     |        \texttt{PAF} (ours) |
| Acceptance\|B          |  $23.12 \pm 16.31$ |   |       $34.54 \pm 0.79$       |   |       $26.65 \pm 1.17$      | $\bf{35.15} \pm \bf{1.36}$ |
| Acceptance\|G          |  $23.13 \pm 16.13$ |   |        $5.43 \pm 0.52$       |   |  $\bf{7.83} \pm \bf{0.52}$  |            $6.04 \pm 0.60$ |
| TCP\|B $\uparrow$      |  $60.57 \pm 42.80$ |   |       $91.70 \pm 1.92$       |   |       $71.51 \pm 3.72$      | $\bf{92.63} \pm \bf{2.08}$ |
| TCP\|G $\uparrow$      |   $69.83 \pm 6.03$ |   |       $69.84 \pm 5.00$       |   |       $53.17 \pm 6.02$      | $\bf{96.74} \pm \bf{1.93}$ |
| FIDiff $\downarrow$    |    $5.55 \pm 6.58$ |   |        $0.97 \pm 0.44$       |   |       $4.42 \pm 0.82$       |  $\bf{0.86} \pm \bf{0.43}$ |
| Accuracy(Y) $\uparrow$ |    $84.50\pm 0.43$ |   |       $92.64 \pm 0.41$       |   | $\bf{98.51} \pm \bf{0.18} $ |           $98.31 \pm 0.26$ |
| Accuracy(G) $\uparrow$ |   $79.25 \pm 9.48$ |   |  $\bf{87.05} \pm \bf{0.46}$  |   |       $86.16 \pm 0.44$      |           $86.67 \pm 0.47$ |
```

(sec:conclusion)= 
## Results \& Discussion
### Analysing the baseline synthetic data
Results comparing the models found in Table {numref}`table:results`.
````{margin}
```{note}
For ease of comparison, we choose to omit the results for `K&C Reweighting` and `Kamishima` from the table and only report `FairLearn`, the model with the highest acceptance percentage for green applicants.
```
````

The baseline data was engineered to demonstrate a biased setting: only $4\%$ of the _green_ candidates are admitted, in comparison to $36\%$ of the _blue_ candidates.
We can evaluate the True Capture percentage (TCP): how many candidates with the ability to graduate, are not being rejected. 
The potential to graduate, $\mathcal{G}$, represents the 'ground truth' potential and correlates directly to $\mathcal{\tilde{X}}$. 
While for blue the TCP is at a high $94\%$, the green TCP is only $60\%$. 
The False Identification difference (FID) measures how well the data conforms to equality of opportunity. 
A low FID, $0.4\%$, means the data conforms to EqOP: once a candidate is accepted, the likelihood of graduation is nearly the same for both groups. 

#### Demographic Parity Oracle
When enforcing demographic parity on the model, the metrics change substantially.
Acceptance percentage is now equal between the _blue_ and _green_ candidates, but at a cost: the TCP for the blue has gone down to $61\%$ and the FID is up to $5.5\%$, meaning there is a higher likelihood of graduating once accepted is no longer independent of $S$.

#### Logistic Regression model
`Logistic Regression` gives similar results to the baseline data. 
We use it as a baseline model for the comparison with subsequent models.  

`FairLearn` achieved the second best acceptance rate for green applicants, after the DP oracle. 
Similar to the DP oracle, however, the additional green applicants who get accepted are not the ones capable of graduating.


#### Positive Action Framework Model
Our `PAF` model shows a $40\%$ improvement in the acceptance rate of green applicants, compared to the engineered baseline data. 
These additional candidates were flagged by our model as falling victim to label bias, i.e. they would have been accepted if they were _perceived_ as blue. 
They are reassigned to 'accept' by the group classifier (selection rule 8 in Table {numref}`table:1`). 
Unlike the DP oracle (`DemPar`) and `FairLearn`, the FID remains under $1\%$, as these additional accepted green applicants are capable of graduating. 
A more notable success, in comparison to the other models, is the high TCP combined with a low FID. 
The addition of the positive action candidate outcome increases the percentage of applicants capable of graduating that are not rejected from $53\%$ to $97\%$. 
This outcome does not come at the expense of equality of opportunity, as the accepted applicants have a similar chance of graduating with a good grade if they were accepted. 

The positive action candidate outcome enables us to not simply reject high-potential candidates from under-represented groups, even if we are not able to accept them under an equal-treatment selection process. 
Equal treatment and equality of opportunity are both maintained for accepted applicants. 

Table {numref}`table:Subgroups` shows the breakdown of the outcome groups, in respect to all the candidates, produced by our `PAF` model. 
This is compared to the outcome groups of a 'perfect' counterfactual (`PCF`). 
The `PCF` is the values we expect if both the encoder and classifier performed without error.
````{margin}
```{note}
These values assume a consistent decision rule across all populations
```
````
We can see that the majority of the candidates receive a counterfactual consensus in both `PCF` and `PAF`. 
`PAF` underestimates the consensus by an overall $7\%$ when compared to the `PCF`, having larger $G_3$ and $G_4$ at the expense of the consensus subgroups. 
In this example, this means the `PCF` assigns rejection for $3\%$ more candidates then the `PAF`.

```{table} Comparison of the number of samples allocated to each group for a ground truth Perfect Counterfactual ($\texttt{PCF}$) in comparison to the Learned Counterfactuals of our Positive Action Framework ($\texttt{PAF}$). We are able to make this comparison only because we know the ground truth for the synthetic data. Our learned model (\texttt{PAF}) is, in general, in agreement with the ground truth (\texttt{PCF}). 
:name: table:Subgroups
| Subgroup | Outcome |       `PCF`      |       `LCF`       |
|----------|:-------:|:----------------:|:-----------------:|
| $g_{1}$  |    1    |  $4.73 \pm 0.15$ |   $3.01 \pm 0.28$ |
| $g_{2}$  |    1    |  $4.73 \pm 0.09$ |   $2.40 \pm 0.48$ |
| $g_{3}$  |    2    | $10.99 \pm 0.19$ |  $13.77 \pm 2.08$ |
| $g_{4}$  |    1    | $10.96 \pm 0.25$ | $15.27 \pm 0.91 $ |
| $g_{5}$  |    0    | $34.35 \pm 0.26$ |  $32.97 \pm 2.86$ |
| $g_{6}$  |    0    | $34.23 \pm 0.28$ |  $32.59 \pm 0.86$ |
```

```{figure} ./assets/adult_figure.png
---
height: 200px
name: fig:adult_results
---
Breakdown of group allocations on the withheld test set of the UCI Adult dataset averaged over 10 repeats, using 3 values as the protected attribute. 
*Left*: The binary 'sex' feature. 
*Middle*: The 'race' feature binarised to membership of the majority group (white). 
*Right*: The 'marital status' feature binarised to whether currently married. 
In all cases, the x-axis represents the percentage of the data that belongs to each protected attribute, while the y-axis represents the percentage of the population assigned each outcome. 
Group membership is defined in Table {numref}`table:1`. 
For all attributes, subgroups $G_{3}$ and $G_{4}$ highlight the proportion of the population for which intervening on the protected attribute will result in the outcome changing as well. 
In $G_{3}$ the outcome changes from negative to positive when $S$ changes, while changes in $G_{4}$ result in the opposite outcome. 
Although an intriguing visualisation of the effect of the different attributes, conclusions should be drawn carefully as the attributes can act as a proxy to hidden patterns in the data. 
Further discussion can be found in Section {numref}`sec:UCI_results`.
```

(sec:UCI_results)=
### Auditing the UCI Adult Income for Bias
Figure {numref}`fig:adult_results` shows a counterfactual subgroup analysis for $3$ protected attributes within the UCI Adult Income data set. 
We note that the accuracy of the PAF model is on par with baseline models. 
As before, the individuals within subgroups $G_{4}$ and $G_{3}$ did not achieve counterfactual consensus. 
For example, for sex, the subgroup $G_{4}$ contains males that are above the $\$50,000$ threshold, but their female counterfactual counterparts would be under the threshold, whereas $G_{3}$ captures females under the threshold whose male counterfactual counterparts would be above the threshold. 
When comparing the effects of the $3$ protected attributes we examined, we can see that changing the marital status is most likely to result in a change the outcome. 
That alone, however, is not enough to deduce a causal relationship between marital status and salary. 
Martial status might have a direct impact on salary but it is also a proxy to other relevant attributes such as age, for example. 
Similarly, the effect of sex can be a combination of a direct effect and a proxy effect from additional relevant attributes such as occupation type and working hours. 
The effect of race may seem to be the least influential of the $3$, but this can be misleading as we can't quantify the contributions of proxy effects on marital status and sex.

Employing our approach on the adult dataset highlights some important challenges and choices: 
Protected attributes may be correlated to other attributes that are relevant to the task. 
These can add proxy effects and mask the direct effect of the protected attribute. 
We can choose to leave the re-calibration of features unrestricted or to keep some features as they were originally. 
The latter may impact the quality of counterfactual representations we can achieve {cite}`PedRugTur08`.

There can be some disparity in the opposite direction to what we expect. 
When comparing males to females in the adult dataset, overall, the direction of bias is in favour of males. 
We do detect, however, females earning above the $\$50,000$ threshold, whose male counterparts would be under the threshold. 
This 'reversed' bias could be present in a subset of occupations.

Under-represented groups can be separated from the majority by a combination of protected attributes. 
Analysing each protected attribute separately does not capture any compounding effects that might be experienced by a specific under-represented group, for example, unmarried, not-white women {cite}`BuoGeb18,FouIslKeyPan20`. 
Considering every possible combination, however, will significantly increase the size of the feature space and as a result, the required size of training data.

```{table} Breakdown of the $G1$ group, comprising individuals funnelled into this group due to different selection rules. Consensus corresponds to selection rule 1 in Table {numref}table:1. Direct bias corresponds to selection rules 2 and 8 in Table~\ref{table:1}. Fallback indicates bias was detected in the opposite direction and the decision reverted to the original outcome.
:name: table:g1
| Selection Rule |   Gender |   Race | Marital Status |
|----------------|---------:|-------:|---------------:|
| Consensus      |   $4.3 $ | $69.2$ |       $ 71.7 $ |
| Direct bias    |   $18.4$ |   $0 $ |            $0$ |
| Fallback       | $ 77.2 $ | $30.8$ |         $28.3$ |
```

### Limitations and Intended Use
When we are considering an algorithmic decision and support system deployed in a real-world biased setting, we can distinguish between different mechanisms that may lead to a disparity in positive outcomes rates between subgroups:
an aggravation of an existing disparity within the training data caused directly by the use of ML, for example, through over-fitting the training data;
bias we can successfully intervene on by mitigating, or even completely removing, label bias from the training data and the learnt model; 
and, a disparity we can detect, but cannot directly intervene on without employing positive discrimination, which is opposed to anti-discrimination legislation.

In this work, we assume we are required to enforce the mapping between the observed and the decision space to be independent of the protected attribute, i.e., we assume it is a requirement to mitigate for label bias (selection rules $2$ \& $8$, Table {numref}`table:1`). 
This is the only bias that is mitigated at the accept / reject level. 
The inclusion of the positive action candidate outcome and the $G_{3}$ subgroup enables us to audit and mitigate, in the form of recommending candidates for positive action, any additional effects that may cause disparity, i.e. selection bias and imbalance of opportunities.  

We choose to adopt a no-detriment, or positive-corrective approach.
This means that no individual, even if they allegedly benefit from past biased decisions, will be made worse off by the positive action approach.
In practice, selection rules can be adapted to suit the context and objectives at hand. 

## Conclusion
We present a novel algorithmic fairness framework that builds on the notion of positive action as a way of advancing equal representation while respecting anti-discrimination legislation and the right for equal treatment. 
We aim not to reject high-potential applicants from under-represented groups, even if they cannot yet successfully compete in an equal-treatment selection process against applicants from the majority group. 
As we are unable to accept them directly, they are highlighted as promising candidates for positive action measures. 

Positive action initiatives can already be found in practice and can include outreach activities, targeted training and adaptive policies. 
Specific positive action measures will be case and context dependent and should be determined by domain experts. 
Our aim is to demonstrate that machine learning has the potential to help identify those applicants who would benefit from this additional support. 

We consider the different mechanisms that can lead to an observed disparity in the rate of positive outcomes between a protected subgroup and the majority. 
We highlight that, at least in part, this disparity can be due to disadvantages affecting applicants belonging to a protected subgroup, hindering their ability to compete with other applicants.

Our counterfactual implementation achieves the goal we set for it: 
it maintains predictive utility while minimising the rejection of candidates with high potential from the disadvantaged group. 
Through this framework we overcome potential challenges in the real-world deployment of decision support systems. 

We hope this work will form part of a larger, constructive discussion, around the role of machine learning in promoting the use and effectiveness of positive action measures.

## Bibliography
```{bibliography}
:filter: docname in docnames
```
