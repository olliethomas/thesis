# Data Domain Fairness

```{math}
\newcommand{\x}{\mathbf{x}}
\newcommand{\xtilde}{\mathbf{\tilde{x}}}
\newcommand{\xhat}{\mathbf{\hat{x}}}
\newcommand{\y}{\mathbf{y}}
\newcommand{\z}{\mathbf{z}}
\newcommand{\rr}{\mathbf{r}}
\newcommand{\w}{\mathbf{w}}
\newcommand{\fhat}{\hat{f}}
\newcommand{\Xcal}{\mathcal{X}}
\newcommand{\Ycal}{\mathcal{Y}}
\newcommand{\Lcal}{\mathcal{L}}
\newcommand{\Gcal}{\mathcal{G}}
\newcommand{\Fcal}{\mathcal{F}}
\newcommand{\Ccal}{\mathcal{C}}
\newcommand{\Zcal}{\mathcal{Z}}
\newcommand{\Rcal}{\mathcal{R}}
\newcommand{\Dcal}{\mathcal{D}}
\newcommand{\PP}{\mathbb{P}} % Probability
\newcommand{\sign}{\text{sign}}
\newcommand{\floor}[1]{\lfloor #1 \rfloor}
\newcommand{\eq}[1]{(\ref{#1})}
\newcommand{\RR}{\mathbb{R}}
\newcommand{\nbr}[1]{\left\|#1\right\|}
\newcommand{\BigO}[1]{\ensuremath{\operatorname{O}\left(#1\right)}}
\newcommand{\Hcal}{\mathcal{H}}
\newcommand{\EE}{\mathbb{E}}
\newcommand{\Eb}{\mathbf{E}}
\newcommand{\inner}[2]{\left\langle #1,#2 \right\rangle}
\def\ci{\perp\!\!\!\perp}
\newcommand{\tr}{\mathop{\mathrm{tr}}}
```

Fair representations are useful, as previously discussed.
But, they are not transparent.


Machine learning systems are increasingly used by government agencies, businesses, and other organisations to assist in making life-changing decisions such as whether or not to invite a candidate to a job interview, or whether to give someone a loan. 
The question is how can we ensure that those systems are _fair_, i.e. they do not discriminate against individuals because of their gender, disability, or other personal ("protected") characteristics?
For example, in building an automated system to review job applications, a photograph might be used in addition to other features to make an invite decision. 
By using the photograph as is, a discrimination issue might arise, as photographs with faces could reveal certain protected characteristics, such as gender, race, or age (e.g. {cite}`FuHeHou2014, BruBurHan93,BroPer93,LevHas15`).
Therefore, any automated system that incorporates photographs into its decision process is at risk of indirectly conditioning on protected characteristics (indirect discrimination).
Recent advances in learning fair representations suggest adversarial training as the means to hide the protected characteristics from the decision/prediction function {cite}`BeuCheZhaChi17,ZhaLemMit18,MadCrePitZem18`. 
All fair representation models, however, learn _latent embeddings_. 
Hence, the produced representations cannot be easily interpreted. 
They do not have the semantic meaning of the input that photographs, or education and training attainments, provide when we have job application data. 
If we want to encourage public conversations and productive public debates regarding fair machine learning systems {cite}`WEF18`, interpretability in how fairness is met is an integral yet overlooked ingredient. 

In this paper we focus on representation learning models that can transform inputs to their fair representations and retain the semantics of the input domain in the transformed space.
When we have image data, our method will make a semantic change to the appearance of an image to deliver a certain fairness criterion [^1]. 
To achieve this, we perform _a data-to-data translation_ by learning a mapping from data in a source domain to a target domain.
Mapping from source to target domain is a standard procedure, and many methods are available. 
For example, in the image domain, if we have aligned source/target as training data, we can use the pix2pix method of {cite}`IsoZhuZhoEfr17`, which is based on conditional generative adversarial networks (cGANs) {cite}`MirOsi14`. 
Zhu et al.'s CycleGAN {cite}`ZhuParIsoEfr17` and Choi et al.'s StarGAN {cite}`ChoChoKimHa18` solve a more challenging setting in which only _un_aligned training examples are available.
However, we can not simply reuse existing methods for source-to-target mapping because we do _not have data in the target domain_ (e.g. fair images are not available; images by themselves can not be fair or unfair, it is only when they are coupled with a particular task that the concern of fairness arises). 

To illustrate the difficulty, consider our earlier example of an automated job review system that uses photographs as part of an input. 
For achieving fairness, it is tempting to simply use GAN-driven methods to _translate female face photos to male_. 
We would require training data of female faces (source domain) and male faces (target domain), and only unaligned training data would be needed. 
This solution is however fundamentally flawed; who gets to decide that we should translate in this direction?
Is it fairer if we translate male faces to female instead?
An ethically grounded approach would be to translate both male and female face photos (source domain) to appropriate middle ground face photos (target domain).
This challenge is actually multi-dimensional, it contains at least _two sub-problems_: a) how to have a general approach that can handle image data as well as tabular data (e.g. work experience, education, or even semantic attribute representations of photographs), and b) how to find a middle-ground with a multi-value (e.g. race) or continuous value (e.g. age) protected characteristic or even multiple characteristics (e.g. race and age).

We propose a solution to the multi-dimensional challenge described above by exploiting statistical (in)dependence between translated images and protected characteristics. 
We use the Hilbert-Schmidt norm of the cross-covariance operator between reproducing kernel Hilbert spaces of image features and protected characteristics (Hilbert-Schmidt independence criterion {cite}`GreBouSmoSch05`) as an empirical estimate of statistical independence.
This flexible measure of independence allows us to take into account higher order independence, and handle a multi-/continuous value and multiple protected characteristics.

## Related work
We focus on expanding the related topic of learning fair, _albeit uninterpretable_, representations.
The aim of fair representation learning is to learn an intermediate representation of the data that preserves as much information about the data as possible, while simultaneously removing protected characteristic information such as age and gender. 
Zemel et al. {cite}`ZemWuSwePitDwo13` learn a probabilistic mapping of the data point to a set of latent prototypes that is independent of protected characteristic (equality of acceptance rates, also called a statistical parity criterion), while retaining as much class label information as possible. 
Louizos et al. {cite}`LouSweLiWelZem15` extend this by employing a deep variational auto-encoder (VAE) framework for finding the fair latent representation. 
In recent years, we see increased adversarial learning methods for fair representations. 
Ganin et al. {cite}`GanUstAjaGerLarLavMarLem16` propose adversarial representation learning for domain adaptation by requiring the learned representation to be indiscriminate with respect to differences in the domains. 
Multiple data domains can be translated into multiple demographic groups.
Edwards and Storkey {cite}`EdwSto16` make this connection and propose adversarial representation learning for the statistical parity criterion. 
To achieve other notions of fairness such as equality of opportunity, Beutel et al. {cite}`BeuCheZhaChi17` show that the adversarial learning algorithm of Edwards and Storkey {cite}`EdwSto16` can be reused but we only supply training data with positive outcome to the adversarial component. 
Madras et al. {cite}`MadCrePitZem18` use a label-aware adversary to learn fair and transferable latent representations for the statistical parity as well as equality of opportunity criteria.

_None of the above_ learn fair representations while simultaneously retaining the semantic meaning of the data. 
There is an orthogonal work on feature selection using human perception of fairness (e.g. {cite}`GrgRedGumWel18`), while this approach undoubtedly retains the semantic meaning of tabular data, it has not been generalized to image data.
In an independent work to ours, Sattigeri et al. {cite}`SatHofCheVar18` describe a similar motivation of producing fair representations in the input image domain; their focus is on creating a whole new image-like dataset, rather than conditioning on each input image.
Hence it is not possible to visualise a fair version for a given image as provided by our method<!--(refer to Figures \ref{fig:faces} and \ref{fig:faces_attributes})-->. 

## Interpretability in Fairness by Residual Decomposition
We will use the illustrative example of an automated job application screening system. 
Given input data (photographs, work experience, education and training, personal skills, etc.) $\x^n \in \Xcal$, output labels of performed well or not well $y^n \in \Ycal = \{+1, -1\}$, and protected characteristic values, such as \emph{race} or \emph{gender}, $s^n \in \{A,B,C,D,\ldots\}$, or \emph{age}, $s^n\in\mathbb{R}$, we would like to train a classifier $f$ that decides whether or not to invite a person for an interview.
We want the classifier to predict outcomes that are accurate with respect to $y^n$ but fair with respect to $s^n$.

(sec:fairnessdefinitions)=
### Fairness definitions
Much work has been done on mathematical definitions of fairness (e.g. {cite}`KleMulRag16,Cho17`). 
It is widely accepted that no single definition of fairness applies in all cases, but will depend on the specific context and application of machine learning models {cite}`WEF18`.
In this paper, we focus on the \emph{equality of opportunity} criterion that requires the classifier $f$ and the protected characteristic $s$ be independent, conditional on the label being positive
\footnote{With binary labels, it is assumed that positive label is a desirable/advantaged outcome, e.g. expected to perform well at the job.}, in shorthand notation $f\ci s\ |\ y = +1$.
Expressing the shorthand notation in terms of a conditional distribution, we have $\PP(f(\x)|s,y=+1) = \PP(f(\x)|y=+1)$.
With binary protected characteristic, this reads as equal true positive rates across the two groups, $\PP(f(\x)=+1|s=A,y=+1)=\PP(f(\x)=+1|s=B,y=+1)$.
Equivalently, the shorthand notation can also be expressed in terms of joint distributions, resulting in $\PP(f(\x),s|y=+1) = \PP(f(\x)|y=+1)\PP(s|y=+1)$.
The advantage of using the joint distribution expression is that the variable $s$ does not appear as a conditioning variable, making it straightforward to use the expression for a multi- or continuous value or even multiple protected characteristics.


### Residual decomposition
We want to learn a data representation $\xtilde^n$ for each input $\x^n$ such that: a) it is able to predict the output label $y^n$, b) it protects $s^n$ according to a certain fairness criterion, c) it lies in the same space as $\x^n$, that is $\xtilde^n\in\Xcal$. 
The third requirement ensures the learned representation to have the same \emph{semantic meaning} as the input. 
For example, for images of people faces, the goal is to modify facial appearance in order to remove the protected characteristic information. 
For tabular data, we desire systematic changes in values of categorical features such as education (bachelors, masters, doctorate, etc.).  
Visualizing those systematic changes will give evidence on how our algorithm enforces a certain fairness criterion.
This will be a powerful tool, albeit all the powers hinge on \emph{observational data}, to scrutinize the interplay between fairness criterion, protected characteristics, and classification accuracy. 
We proceed by making the following decomposition assumption on $\x$:
```{math}
:label: eq:decomposition
\phi(\x) = \phi(\xtilde) + \phi(\xhat)
```
with $\xtilde$ to be the component that is independent of $s$, $\xhat$ denoting the component of $\x$ that is dependent on $s$, and $\phi(\cdot)$ is some \emph{pre-trained} feature map. 
We will discuss about the specific choice of this pre-trained feature map for both image and tabular data later in the section.
What we want is to learn a mapping from a source domain (input features) to a target domain (fair features with the semantics of the input domain), i.e.
T: \x \rightarrow \xtilde,  
and we will parameterize this mapping $T = T_{\omega}$ where $\omega$ is a class of autoencoding transformer network. 
For our architectural choice of transformer network, please refer to Section \ref{sec:experiments}.

To enforce the decomposition structure in \eq{eq:decomposition}, we need to satisfy two conditions: a) $\xtilde$ to be independent of $s$, and b) $\xhat$ to be dependent of $s$. 
Given a particular statistical dependence measure, the first condition can be achieved by \emph{minimizing} the dependence measure between $P = \{\phi(\xtilde^{1}),\ldots,\phi(\xtilde^{N})\} = \{\phi(T_{\omega}(\x^1)),\ldots,\phi(T_{\omega}(\x^N))\}$ and $S=\{s^1,\ldots,s^N\}$; $N$ is the number of training data points. 
For the second condition, we first define a \emph{residual}:
```{math}
\phi(\x) - \phi(\xtilde) = \phi(\x) - \phi(T_{\omega}(\x)) = \phi(\xhat)
```
where the last term is the data component that is \emph{dependent} on a protected characteristic $s$.
We can then enforce the second condition by \emph{maximizing} the dependence measure between $R = \{\phi(\xhat^{1}),\ldots,\phi(\xhat^{N})\} = \{\phi(\x^1) - \phi(T_{\omega}(\x^1)),\ldots,\phi(\x^N) - \phi(T_{\omega}(\x^N))\}$ and $S$. 
We use the decomposition property as a guiding mechanism to learn the parameters $\omega$ of the transformer network $T_{\omega}$.

In the fair and interpretable representation learning task, we believe using residual is well-motivated because we know that our generated fair features should be somewhat similar to our input features.
Residuals will make learning the transformer network easier. 
Taking into consideration that we do not have training data about the target fair features $\xtilde$, we should not desire the transformer network to take the input feature $\x$ and \emph{generate} a new output $\xtilde$.
Instead, it should just learn how to \emph{adjust} our input $\x$ to produce the desired output $\xtilde$.
The concept of residuals is universal, for example, a residual block has been used to speed up and to prevent over-fitting of a very deep neural network {cite}`HeZhaRenSun16`, and a residual regression output has been used to perform causal inference in additive noise models {cite}`MooJanPetSch09`.

Formally, given the $N$ training triplets $(X,S,Y)$, to find a fair and interpretable representation $\xtilde= T_{\omega}(\x)$, our optimization problem is given by:
```{math}
:label: eq:optproblem
\begin{align}
  &  \min_{T_\omega}   \underbrace{\sum_{n=1}^N\Lcal(T_{\omega}(\x^n),y^n)}_{\text{prediction loss}} + \lambda_1 \underbrace{\sum_{n=1}^{N}\|\x^n-T_{\omega}(\x^n)\|_2^2}_{\text{reconstruction loss}} + \nonumber\\
  &+ \lambda_2\left(\underbrace{- \text{HSIC}(R,S|Y=+1) + \text{HSIC}(P,S|Y=+1)}_{\text{decomposition loss}}\right) 
\end{align}
``` 
where $\text{HSIC}(\cdot,\cdot)$ is the statistical dependence measure, and $\lambda_i$ are trade-off parameters. 
HSIC is the Hilbert-Schmidt norm of the cross-covariance operator between reproducing kernel Hilbert spaces.
This is equivalent to a non-parametric distance measure of a joint distribution and the product of two marginal distributions using the Maximum Mean Discrepancy (MMD) criterion{cite}`GreBorRasSchetal12`; MMD has been successfully used in fairnesss literature in it's own right {cite}`LouSweLiWelZem15, QuaSha17`. Section \ref{sec:fairnessdefinitions} discusses defining statistical independence based on a joint distribution, contrasting this with a conditional distribution.
We use the biased estimator of HSIC {cite}`GreBouSmoSch05,SonSmoGreBedetal12`: $\text{HSIC}_{\text{emp.}} = (N-1)^{-2}\tr HKHL,$
where $K, L \in\mathbb{R}^{N\times N}$ are the kernel matrices for the \emph{residual} set $R$ and the protected characteristic set $S$ respectively, i.e.\ $K_{ij} = k(r^i, r^j)$ and $L_{ij} = l(s^i, s^j)$ (similar definition for measuring independence between sets $P$ and $S$). 
We use a Gaussian RBF kernel function for both $k(\cdot,\cdot)$ and $l(\cdot,\cdot)$. 
Moreover, $H_{ij}=\delta_{ij}-N^{-1}$ centres the observations of set $R$ and set $S$ in RKHS feature space. 
The prediction loss is defined using a softmax layer on the output of the transformer network. 
While in image data we add the total variation (TV) penalty {cite}`MahVed15` on the fair representation to ensure spatial smoothness, we do not enforce any regularization term for tabular data. 

\noindent In summary, we learn a new representation $\xtilde{}$ that removes statistical dependence on the protected characteristic $s$ (by minimizing $\text{HSIC}(P,S|Y=+1)$) and enforces the dependence of the residual $\x-\xtilde$ and $s$ (by maximizing $\text{HSIC}(R,S|Y=+1)$). %For equality of opportunity, we ensure the conditional independence criterion $(\xtilde{} \ci{} s)|y=+1$. 
We can then train any classifier $f$ using this new representation, and it will inherently satisfy the fairness criterion {cite}`MadCrePitZem18`.

### Hilbert-Schmidt independence criterion
We could use mutual information as the statistical dependence measure, however, it has been shown that 
computing mutual information in high dimensions (our pre-trained feature map $\phi(\cdot)$ is high dimensional) requires sophisticated bias correction methods {cite}`NemShaBia02`.
Instead of mutual information, we use the Hilbert-Schmidt Independence Criterion (HSIC) {cite}`GreBouSmoSch05` as our measure of statistical dependence in \eq{eq:optproblem}, i.e. $\text{Dep}(\cdot,\cdot) = \text{HSIC}$.
HSIC is the Hilbert-Schmidt norm of the cross-covariance operator between reproducing kernel Hilbert spaces.
This is equivalent to a non-parametric distance measure of a joint distribution and the product of two marginal distributions using the Maximum Mean Discrepancy (MMD) criterion {cite}`GreBorRasSchetal12` (which has been successfully used in fairnesss literature in it's own right {cite}`LouSweLiWelZem15, QuaSha17`). Section \ref{sec:fairnessdefinitions} discusses defining statistical independence based on a joint distribution, contrasting this with a conditional distribution.
HSIC has several advantages: first, it does not require density estimation, and second, it has very little bias, even in high dimensions.
Given a sample
$Z=\{(r^1,s^1),\ldots,(r^N,s^N)\}$ of size $N$ drawn from $\PP_{rs}$ an
empirical estimate of HSIC is given by
```{math}
:label: eq:e_hsic
\text{HSIC}_{\text{emp.}} = (N-1)^{-2}\tr HKHL = (N-1)^{-2} \tr \bar{K} \bar{L}
```
where $K, L \in\mathbb{R}^{N\times N}$ are the kernel matrices for the \emph{residual}
set $R$ and the protected characteristic set $S$ respectively, i.e.\ $K_{ij} = k(r^i, r^j)$ and
$L_{ij} = l(s^i, s^j)$ (similar definition for measuring independence between sets $P$ and $S$). 
We use a Gaussian RBF kernel function for both $k(\cdot,\cdot)$ and $l(\cdot,\cdot)$. 
Moreover, $H_{ij}=\delta_{ij}-N^{-1}$ centres the observations of set $R$ and set $S$ in RKHS feature space. 
Finally, $\bar{K} := H K H$ and $\bar{L} := H L H$ denote the centred versions of $K$ and $L$ respectively. 
For complete statistical properties of the empirical estimator in \eq{eq:e_hsic} refer to {cite}`GreBouSmoSch05`.

### Neural style transfer and pre-trained feature space
Neural style transfer (e.g. {cite}`GatEckBet15a,JohAlaFei2016`) is a popular approach to perform an image-to-image translation.
Our decomposition loss in \eq{eq:optproblem} is reminiscent of a style loss used in neural style transfer models.
The style loss is defined as the distance between second-order statistics of a style image and the translated image.
Excellent results {cite}`GatEckBet15a,JohAlaFei2016,UlyLebVedLem16,ulyanov2017` on neural style transfer rely on pre-trained features.
Following this spirit, we also use a ``pre-trained'' feature mapping $\phi(\cdot)$ in defining our decomposition loss.
For image data, we take advantage of the powerful representation of deep convolutional neural networks (CNN) to define the mapping function {cite}`GatEckBet15a`.
The feature maps of $\x$ in the layer $l$ of a CNN are denoted by $F^l_\x\in R^{N_l\times M_l}$ where $N_l$ is the number of the feature maps in the layer $l$ and $M_l$ is the height times the width of the feature map.
We use the vectorization of $F^l_\x$ as the required mapping $\phi(\x) = \text{vec}(F^l_\x)$.
Several layers of a CNN will be used to define the full mapping (see Section \ref{sec:experiments}). 
For tabular data, we use the following random Fourier feature {cite}`RahRec08` mapping $\phi(\x) = \sqrt{\frac{2}{D}}\ \text{cos}(\inner{\theta}{\x}+b)$ with a bias vector $b\in\RR^D$ that is uniformly sampled in $[0,2\pi]$, and a matrix 
$\theta\in\RR^{d\times D}$ where $\theta_{ij}$ is sampled from a Gaussian distribution. 
We have assumed the input data lies in a $d$-dimensional space, and we transform them to a $D$-dimensional space.





[^1]: Examples of fairness criteria are equality of true positive rates (TPR), also called equality of opportunity {cite}`HarPriSre16,ZafValRodGum17b`, between males and females.


---
```{bibliography}
:filter: docname in docnames
```
