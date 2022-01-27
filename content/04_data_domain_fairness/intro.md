(ch:paper2)=
# Chapter 5: Data Domain Fairness

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

#### TODO

- [ ] reconstruction losses are generally at odds with fair representations.
- [ ] The need for zs/zy partitions


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
Given input data (photographs, work experience, education and training, personal skills, etc.) $\x^n \in \Xcal$, output labels of performed well or not well $y^n \in \Ycal = \{+1, -1\}$, and protected characteristic values, such as _race_ or _gender_, $s^n \in \{A,B,C,D,\ldots\}$, or _age_, $s^n\in\mathbb{R}$, we would like to train a classifier $f$ that decides whether or not to invite a person for an interview.
We want the classifier to predict outcomes that are accurate with respect to $y^n$ but fair with respect to $s^n$.

(sec:fairnessdefinitions)=
### Fairness definitions
Much work has been done on mathematical definitions of fairness (e.g. {cite}`KleMulRag17,Cho17`). 
It is widely accepted that no single definition of fairness applies in all cases, but will depend on the specific context and application of machine learning models {cite}`WEF18`.
In this paper, we focus on the _equality of opportunity_ criterion that requires the classifier $f$ and the protected characteristic $s$ be independent, conditional on the label being positive [^2], in shorthand notation $f\ci s\ |\ y = +1$.
Expressing the shorthand notation in terms of a conditional distribution, we have $\PP(f(\x)|s,y=+1) = \PP(f(\x)|y=+1)$.
With binary protected characteristic, this reads as equal true positive rates across the two groups, $\PP(f(\x)=+1|s=A,y=+1)=\PP(f(\x)=+1|s=B,y=+1)$.
Equivalently, the shorthand notation can also be expressed in terms of joint distributions, resulting in $\PP(f(\x),s|y=+1) = \PP(f(\x)|y=+1)\PP(s|y=+1)$.
The advantage of using the joint distribution expression is that the variable $s$ does not appear as a conditioning variable, making it straightforward to use the expression for a multi- or continuous value or even multiple protected characteristics.


### Residual decomposition
We want to learn a data representation $\xtilde^n$ for each input $\x^n$ such that: a) it is able to predict the output label $y^n$, b) it protects $s^n$ according to a certain fairness criterion, c) it lies in the same space as $\x^n$, that is $\xtilde^n\in\Xcal$. 
The third requirement ensures the learned representation to have the same _semantic meaning_ as the input. 
For example, for images of people faces, the goal is to modify facial appearance in order to remove the protected characteristic information. 
For tabular data, we desire systematic changes in values of categorical features such as education (bachelors, masters, doctorate, etc.).  
Visualizing those systematic changes will give evidence on how our algorithm enforces a certain fairness criterion.
This will be a powerful tool, albeit all the powers hinge on _observational data_, to scrutinize the interplay between fairness criterion, protected characteristics, and classification accuracy. 
We proceed by making the following decomposition assumption on $\x$:
```{math}
:label: eq:decomposition
\phi(\x) = \phi(\xtilde) + \phi(\xhat)
```
with $\xtilde$ to be the component that is independent of $s$, $\xhat$ denoting the component of $\x$ that is dependent on $s$, and $\phi(\cdot)$ is some _pre-trained_ feature map. 
We will discuss about the specific choice of this pre-trained feature map for both image and tabular data later in the section.
What we want is to learn a mapping from a source domain (input features) to a target domain (fair features with the semantics of the input domain), i.e.
T: \x \rightarrow \xtilde,  
and we will parameterize this mapping $T = T_{\omega}$ where $\omega$ is a class of autoencoding transformer network. 
For our architectural choice of transformer network, please refer to Section \ref{sec:experiments}.

To enforce the decomposition structure in \eq{eq:decomposition}, we need to satisfy two conditions: a) $\xtilde$ to be independent of $s$, and b) $\xhat$ to be dependent of $s$. 
Given a particular statistical dependence measure, the first condition can be achieved by _minimizing_ the dependence measure between $P = \{\phi(\xtilde^{1}),\ldots,\phi(\xtilde^{N})\} = \{\phi(T_{\omega}(\x^1)),\ldots,\phi(T_{\omega}(\x^N))\}$ and $S=\{s^1,\ldots,s^N\}$; $N$ is the number of training data points. 
For the second condition, we first define a _residual_:
```{math}
\phi(\x) - \phi(\xtilde) = \phi(\x) - \phi(T_{\omega}(\x)) = \phi(\xhat)
```
where the last term is the data component that is _dependent_ on a protected characteristic $s$.
We can then enforce the second condition by _maximizing_ the dependence measure between $R = \{\phi(\xhat^{1}),\ldots,\phi(\xhat^{N})\} = \{\phi(\x^1) - \phi(T_{\omega}(\x^1)),\ldots,\phi(\x^N) - \phi(T_{\omega}(\x^N))\}$ and $S$. 
We use the decomposition property as a guiding mechanism to learn the parameters $\omega$ of the transformer network $T_{\omega}$.

In the fair and interpretable representation learning task, we believe using residual is well-motivated because we know that our generated fair features should be somewhat similar to our input features.
Residuals will make learning the transformer network easier. 
Taking into consideration that we do not have training data about the target fair features $\xtilde$, we should not desire the transformer network to take the input feature $\x$ and _generate_ a new output $\xtilde$.
Instead, it should just learn how to _adjust_ our input $\x$ to produce the desired output $\xtilde$.
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
This is equivalent to a non-parametric distance measure of a joint distribution and the product of two marginal distributions using the Maximum Mean Discrepancy (MMD) criterion{cite}`GreBorRasSchSmo12`; MMD has been successfully used in fairnesss literature in it's own right {cite}`LouSweLiWelZem15, QuaSha17`. Section \ref{sec:fairnessdefinitions} discusses defining statistical independence based on a joint distribution, contrasting this with a conditional distribution.
We use the biased estimator of HSIC {cite}`GreBouSmoSch05,SonSmoGreBedBor12`: $\text{HSIC}_{\text{emp.}} = (N-1)^{-2}\tr HKHL,$
where $K, L \in\mathbb{R}^{N\times N}$ are the kernel matrices for the _residual_ set $R$ and the protected characteristic set $S$ respectively, i.e.\ $K_{ij} = k(r^i, r^j)$ and $L_{ij} = l(s^i, s^j)$ (similar definition for measuring independence between sets $P$ and $S$). 
We use a Gaussian RBF kernel function for both $k(\cdot,\cdot)$ and $l(\cdot,\cdot)$. 
Moreover, $H_{ij}=\delta_{ij}-N^{-1}$ centres the observations of set $R$ and set $S$ in RKHS feature space. 
The prediction loss is defined using a softmax layer on the output of the transformer network. 
While in image data we add the total variation (TV) penalty {cite}`MahVed15` on the fair representation to ensure spatial smoothness, we do not enforce any regularization term for tabular data. 

\noindent In summary, we learn a new representation $\xtilde{}$ that removes statistical dependence on the protected characteristic $s$ (by minimizing $\text{HSIC}(P,S|Y=+1)$) and enforces the dependence of the residual $\x-\xtilde$ and $s$ (by maximizing $\text{HSIC}(R,S|Y=+1)$). 
We can then train any classifier $f$ using this new representation, and it will inherently satisfy the fairness criterion {cite}`MadCrePitZem18`.

### Hilbert-Schmidt independence criterion
We could use mutual information as the statistical dependence measure, however, it has been shown that 
computing mutual information in high dimensions (our pre-trained feature map $\phi(\cdot)$ is high dimensional) requires sophisticated bias correction methods {cite}`NemShaBia02`.
Instead of mutual information, we use the Hilbert-Schmidt Independence Criterion (HSIC) {cite}`GreBouSmoSch05` as our measure of statistical dependence in \eq{eq:optproblem}, i.e. $\text{Dep}(\cdot,\cdot) = \text{HSIC}$.
HSIC is the Hilbert-Schmidt norm of the cross-covariance operator between reproducing kernel Hilbert spaces.
This is equivalent to a non-parametric distance measure of a joint distribution and the product of two marginal distributions using the Maximum Mean Discrepancy (MMD) criterion {cite}`GreBorRasSchSmo12` (which has been successfully used in fairnesss literature in it's own right {cite}`LouSweLiWelZem15, QuaSha17`). Section \ref{sec:fairnessdefinitions} discusses defining statistical independence based on a joint distribution, contrasting this with a conditional distribution.
HSIC has several advantages: first, it does not require density estimation, and second, it has very little bias, even in high dimensions.
Given a sample
$Z=\{(r^1,s^1),\ldots,(r^N,s^N)\}$ of size $N$ drawn from $\PP_{rs}$ an
empirical estimate of HSIC is given by
```{math}
:label: eq:e_hsic
\text{HSIC}_{\text{emp.}} = (N-1)^{-2}\tr HKHL = (N-1)^{-2} \tr \bar{K} \bar{L}
```
where $K, L \in\mathbb{R}^{N\times N}$ are the kernel matrices for the _residual_
set $R$ and the protected characteristic set $S$ respectively, i.e.\ $K_{ij} = k(r^i, r^j)$ and
$L_{ij} = l(s^i, s^j)$ (similar definition for measuring independence between sets $P$ and $S$). 
We use a Gaussian RBF kernel function for both $k(\cdot,\cdot)$ and $l(\cdot,\cdot)$. 
Moreover, $H_{ij}=\delta_{ij}-N^{-1}$ centres the observations of set $R$ and set $S$ in RKHS feature space. 
Finally, $\bar{K} := H K H$ and $\bar{L} := H L H$ denote the centred versions of $K$ and $L$ respectively. 
For complete statistical properties of the empirical estimator in \eq{eq:e_hsic} refer to {cite}`GreBouSmoSch05`.

### Neural style transfer and pre-trained feature space
Neural style transfer (e.g. {cite}`GatEckBat16,JohAlaFei2016`) is a popular approach to perform an image-to-image translation.
Our decomposition loss in \eq{eq:optproblem} is reminiscent of a style loss used in neural style transfer models.
The style loss is defined as the distance between second-order statistics of a style image and the translated image.
Excellent results {cite}`GatEckBat16,JohAlaFei2016,UlyLebVedLem16,ulyanov2017` on neural style transfer rely on pre-trained features.
Following this spirit, we also use a "pre-trained" feature mapping $\phi(\cdot)$ in defining our decomposition loss.
For image data, we take advantage of the powerful representation of deep convolutional neural networks (CNN) to define the mapping function {cite}`GatEckBat16`.
The feature maps of $\x$ in the layer $l$ of a CNN are denoted by $F^l_\x\in R^{N_l\times M_l}$ where $N_l$ is the number of the feature maps in the layer $l$ and $M_l$ is the height times the width of the feature map.
We use the vectorization of $F^l_\x$ as the required mapping $\phi(\x) = \text{vec}(F^l_\x)$.
Several layers of a CNN will be used to define the full mapping (see Section \ref{sec:experiments}). 
For tabular data, we use the following random Fourier feature {cite}`RahRec08` mapping $\phi(\x) = \sqrt{\frac{2}{D}}\ \text{cos}(\inner{\theta}{\x}+b)$ with a bias vector $b\in\RR^D$ that is uniformly sampled in $[0,2\pi]$, and a matrix 
$\theta\in\RR^{d\times D}$ where $\theta_{ij}$ is sampled from a Gaussian distribution. 
We have assumed the input data lies in a $d$-dimensional space, and we transform them to a $D$-dimensional space.

(sec:experiments)=
## Experiments

We gave an illustrative example about screening job applications, however, no such data is publicly available. 
We will instead use publicly available data to simulate the setting. 
We conduct the experiments using three datasets: the [CelebA image dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) {cite}`LiuLuoWanTan15` , the [Diversity in Faces (DiF) dataset](https://www.research.ibm.com/artificial-intelligence/trusted-ai/diversity-in-faces/) {cite}`DiF2019`, and the [Adult income dataset](https://archive.ics.uci.edu/ml/datasets/adult) from the UCI repository {cite}`Dua:2017`. 
The CelebA dataset has a total of $202,599$ celebrity images. The images are annotated with $40$ attributes that reflect appearance (hair color and style, face shape, makeup, for example), emotional state (smiling), gender, attractiveness, and age. For this dataset, we use gender as a binary protected characteristic, and attractiveness as the proxy measure of getting invited for a job interview in the world of fame. 
We randomly select $20$K images for testing and use the rest for training the model.
The DiF dataset has only been introduced very recently and contains nearly a million human face images reflecting diversity in ethnicity, age and gender. 
We include preliminary results using 200K images for training and 200K images for testing our model on this dataset. The images are annotated with attributes such as race, gender and age (both continual and discretized into seven age groups) as well as facial landmarks and facial symmetry features. For this dataset, we use gender as a binary protected characteristic, and the discretized age groups as a predictive task. 
The Adult income dataset is frequently used to assess fairness methods. It comes from the Census bureau and the binary task is to predict whether or not an individual earns more than $\$ 50$K per year. It has a total of $45,222$ data instances, each with $14$ features such as gender, marital status, educational level, number of work hours per week. 
For this dataset, we follow {cite}`ZemWuSwePitDwo13` and consider gender as a binary protected characteristic. 
We use $28,222$ instances for training, and $15,000$ instances for testing. 
We enforce equality of opportunity as the fairness criteria throughout for the three experiments.

```{table} Results of training multiple classifiers (rows 1--7) on $3$ different representations, $\x$, $\xtilde$, and $\mathbf{z}$. $\x$ is the original input representation, $\xtilde$ is the interpretable, fair representation introduced in this paper, and $\mathbf{z}$ is the latent embedding representation of Beutel et al. We **boldface** Eq. Opp. since this is the fairness criterion (the lower the better). $^*$The solver of $\texttt{Zafar et al.}$ fails to converge in 4 out of 10 repeats. Our learned representation $\xtilde$ achieves comparable fairness level to the latent representation $\mathbf{z}$, while maintaining the constraint of being in the same space as the original input.
:name: table:benchmarking
|                                                       |    original $\x$    |                       | fair interpretable $\xtilde$ |                      | latent embedding $z$ |                      |
|-------------------------------------------------------|:-------------------:|----------------------:|:----------------------------:|:--------------------:|:--------------------:|:--------------------:|
|                                                       | Accuracy $\uparrow$ |  Eq. Opp $\downarrow$ |      Accuracy $\uparrow$     | Eq. Opp $\downarrow$ |  Accuracy $\uparrow$ | Eq. Opp $\downarrow$ |
| 1: $\texttt{LR}$                                      |     $85.1\pm0.2$    |  $\mathbf{9.2\pm2.3}$ |         $84.2\pm0.3$         | $\mathbf{5.6\pm2.5}$ |     $81.8\pm2.1$     | $\mathbf{5.9\pm4.6}$ |
| 2: $\texttt{SVM}$                                     |     $85.1\pm0.2$    |  $\mathbf{8.2\pm2.3}$ |         $84.2\pm0.3$         | $\mathbf{4.9\pm2.8}$ |     $81.9\pm2.0$     | $\mathbf{6.7\pm4.7}$ |
| 3: $\texttt{Fair Reduction LR}${cite}`AgaBeyDudLanWal18`          |     $85.1\pm0.2$    | $\mathbf{14.9\pm1.3}$ |         $84.1\pm0.3$         | $\mathbf{6.5\pm3.2}$ |     $81.8\pm2.1$     | $\mathbf{5.6\pm4.8}$ |
| 4: $\texttt{Fair Reduction SVM}${cite}`AgaBeyDudLanWal18`        |     $85.1\pm0.2$    |  $\mathbf{8.2\pm2.3}$ |         $84.2\pm0.3$         | $\mathbf{4.9\pm2.8}$ |     $81.9\pm2.0$     | $\mathbf{6.7\pm4.7}$ |
| 5: $\texttt{Kamiran \& Calders LR}${cite}`KamCal12`  |     $84.4\pm0.2$    | $\mathbf{14.9\pm1.3}$ |         $84.1\pm0.3$         | $\mathbf{1.7\pm1.3}$ |     $81.8\pm2.1$     | $\mathbf{4.9\pm3.3}$ |
| 6: $\texttt{Kamiran \& Calders SVM}${cite}`KamCal12` |     $85.1\pm0.2$    |  $\mathbf{8.2\pm2.3}$ |         $84.2\pm0.3$         | $\mathbf{4.9\pm2.8}$ |     $81.9\pm2.0$     | $\mathbf{6.7\pm4.7}$ |
| 7: $\texttt{Zafar et al.}^*${cite}`ZafValRodGum17b`  |     $85.0\pm0.3$    |  $\mathbf{1.8\pm0.9}$ |              ---             |          ---         |          ---         |          ---         |
```

\begin{figure*}[tb]
\begin{center}
\scalebox{0.95}{
\begin{tabular}{c|c}
  \includegraphics[width=0.24\textwidth]{figures/relationship_x_compressed.pdf}\includegraphics[width=0.24\textwidth]{figures/relationship_x_tilde_compressed.pdf}  & 
  \includegraphics[width=0.24\textwidth]{figures/race_x_compressed.pdf}\includegraphics[width=0.24\textwidth]{figures/race_x_tilde_compressed.pdf}\\
  (`Relationship Status`) & (`Race`)
\end{tabular}
}
\caption{**Left** Boxplots showing the distribution of the categorical feature `Relationship Status` **Right** Boxplots showing the distribution of the categorical feature `Race`.
**Left of each**: original representation $\x\in\Xcal$. **Right of each**: fair representation $\xtilde\in\Xcal$.\label{fig:interpretability}}
\end{center}
\end{figure*}

### The Adult Income dataset
The focus is to investigate whether (**Q1**) our proposed fair and interpretable learning method performs on a par with state-of-the-art fairness methods, and whether (**Q2**) performing a tabular-to-tabular translation brings us closer to achieving interpretability in how fairness is being satisfied.
We compare our method against an unmodified $\x{}$ using the following classifiers: 
1) logistic regression (`LR`) and 
2) support vector machine with linear kernel (`SVM`),
We select the regularization parameter of `LR` and `SVM` over 6 possible values ($10^i$ for $i \in [0,6]$) using $3$-fold cross validation.
We then train classifiers 1--2 with the learned representation $\xtilde{}$ and with the latent embedding $\z{}$ of a state-of-the-art adversarial model described in Beutel et al. {cite}`BeuCheZhaChi17`. 
We also apply methods which reweigh the samples to simulate a balanced dataset with regard to the protected characteristic FairLearn {cite}`AgaBeyDudLanWal18` `Fair Reduction` 3-4 and Kamiran \& Calders {cite}`KamCal12` `Kamiran & Calders` 5-6, 
optimized with both the cross-validated `LR` and `SVM` (1-2), 
giving (`Fair Reduction LR`), (`Fair Reduction SVM`), (`Kamiran & Calders LR`) and (`Kamiran & Calders SVM`) respectively.
As a reference, we also compare with:
7) Zafar et al.'s{cite}`ZafValRodGum17b` fair classification method (`Zafar et al.`) that adds equality of opportunity directly as a constraint to the learning objective function.
It has been shown that applying fairness constraints in succession as 'fair pipelines' do not enforce fairness {cite}`DwoIlv18, BowKitNisStraVarVen17`, as such, we only demonstrate (fair) classifier 7 on the unmodified $\x{}$.


#### Benchmarking
We train our model for $50,000$ iterations using a network with 1 hidden layer of $40$ nodes for both the encoder and decoder, with the encoded representation being 40 nodes. The predictor acts on the decoded output of this network. We set the trade-off parameters of the reconstruction loss ($\lambda_1$) and decomposition loss ($\lambda_2$) to $10^{-4}$ and $100$ respectively.
We then use this model to translate $10$ different training and test sets into $\xtilde{}$.
Using a modified version of the framework provided by Friedler et al. {cite}`FrieSchVen18` we evaluate methods $1$--$6$ using $\x{}$ and $\xtilde{}$ representations. To ensure consistency, we train the model of Beutel et al. {cite}`BeuCheZhaChi17` with the same architecture and number of iterations as our model.

Table \ref{table:benchmarking} shows the results of these experiments. Our interpretable representation, $\xtilde{}$ achieves similar fairness level to Beutel's state-of-the-art approach (**Q1**). Consistently, our representation $\xtilde{}$ promoted the _fairness_ criterion (Eq. Opp. close to $0$), with only a small penalty in accuracy. 

#### Interpretability
We promote equality of opportunity for the positive class ($\text{actual salary} > \text{\$50K}$). 
In Figure \ref{fig:interpretability} we show the effect of learning a fair representation, showing changes in the `Relationship Status` and `Race` features of samples that were incorrectly classified by an SVM as earning $<$ \$50K in $\x{}$, but were correctly classified in $\xtilde{}$. 
The visualization can be used for understanding how representation methods adjust the data for fairness. 
For example in Figure \ref{fig:interpretability} (left) we can see that our method deals with the notorious problem of a husband or wife relationship status being a direct proxy for gender (**Q2**). Our method recognises this across all repeats in an unsupervised manner and reduces the wife category which is associated with a negative prediction. Other categories that have less correlation with the protected characteristic, such as race, largely remain unmodified (Figure \ref{fig:interpretability} (right)). 

### The CelebA dataset
Our intention here is to investigate whether (**Q3**) performing an image-to-image translation brings us closer to achieving interpretability in how fairness is being satisfied, and whether (**Q4**) using semantic attribute representations of images reinforces similar interpretability conclusions as using image features directly.

\begin{figure}[t]
\begin{tabular}{l}
\hspace{-0.2cm}translated\hspace{-0.4cm}
\end{tabular}
\begin{tabular}{l}
\includegraphics[width=0.095\textwidth]{figures/celeba_res/006126.jpg} 
\includegraphics[width=0.095\textwidth]{figures/celeba_res/015365.jpg} 
\includegraphics[width=0.095\textwidth]{figures/celeba_res/015505.jpg} 
\includegraphics[width=0.095\textwidth]{figures/celeba_res/028255.jpg} 
\end{tabular}\\
\begin{tabular}{l}
\hspace{-0.2cm}residual\hspace{-0.15cm}  
\end{tabular}
\begin{tabular}{l}
\includegraphics[width=0.095\textwidth]{figures/celeba_res/006126res.jpg}
\includegraphics[width=0.095\textwidth]{figures/celeba_res/015365res.jpg}
\includegraphics[width=0.095\textwidth]{figures/celeba_res/015505res.jpg}
\includegraphics[width=0.095\textwidth]{figures/celeba_res/028255res.jpg}
\end{tabular}
\caption{Examples of the translated and residual images on CelebA from the protected group of males (minority group) that have been classified correctly (as attractive) after transformation. These results are obtained with the transformer network for image-to-image translation.
Best viewed in color.}
\label{fig:faces}
\end{figure}

```{table} Results on CelebA dataset using a variety of input domains. Prediction performance is measured by accuracy, and we use equality of opportunity, TPRs difference, as the fairness criterion. Here, domain of fake images (last row) denotes images synthesized by the StarGAN model from the original images and their fair attribute representations. We **boldface** Eq. Opp. since this is the fairness criterion.
:name: tab:results_celeba
|                |       domain       |    Acc.    |     Eq. Opp.    |        TPR        |       TPR       |
|:--------------:|:------------------:|:----------:|:---------------:|:-----------------:|:---------------:|
|                |       $\Xcal$      | $\uparrow$ |   $\downarrow$  | ${\text{female}}$ | ${\text{male}}$ |
|   orig. $\x $  |    \emph{images}   |   $80.6$   | $\mathbf{33.8}$ |       $90.8$      |      $57.0$     |
|   orig. $\x $  |  \emph{attributes} |   $79.1$   | $\mathbf{39.9}$ |       $90.8$      |      $50.9$     |
| fair $\xtilde$ |    \emph{images}   |   $79.4$   | $\mathbf{23.8}$ |       $85.2$      |      $61.4$     |
| fair $\xtilde$ |  \emph{attributes} |   $75.9$   | $\mathbf{12.4}$ |       $87.2$      |      $74.8$     |
| fair $\xtilde$ | \emph{fake images} |   $78.5$   | $\mathbf{23.0}$ |       $87.5$      |      $64.5$     |
```

#### Image-to-image translation 
Our autoencoder network is based on the architecture of the transformer network for neural style transfer {cite}`JohAlaFei2016` with three convolutional layers, five residual layers and three deconvolutional/upsampling layers in combination with instance weight normalization {cite}`ulyanov2017`. 
The transformer network produces the residual image using a non-linear tanh activation, which is then subtracted from the input image to form the translated fair image $\xtilde$.
Similarly to neural style transfer {cite}`GatEckBat16, gardner2016, JohAlaFei2016`, for computing the loss terms, we use the activations in the deeper layers of the 19-layered VGG19 network {cite}`SimZis15` as feature representations of both input and translated images. Specifically, we use activations in the conv3\_1, conv4\_1 and conv5\_1 layers for computing the decomposition loss, the conv3\_1 layer activations for the reconstruction loss, and the activations in the last convolutional layer pool\_5 for the prediction loss and when evaluating the performance. Given a 176x176 color input image, we compute the activations at each layer mentioned earlier after ReLU, then we flatten and $l_2$ normalize them to form features for the loss terms.
In the HSIC estimates of the decomposition loss, we use a Gaussian RBF kernel $k(x_1,x_2) = \text{exp} (-\gamma \|x_1-x_2\|^2)$ width $\gamma=1.0$ for image features, and $\gamma =0.5$ for protected characteristics (as one over squared distance in the binary space).
To compute the decomposition loss, we add the contributions across the three feature layers. 
We set the trade-off parameters $\lambda_1$ and $\lambda_2$ of the reconstruction loss and the decomposition loss, respectively, to $1.0$, and the TV regularization strength, $\lambda_3$ 
to $10^{-3}$. 
Training was carried out for 50 epochs with a batch size of $80$ images.
We use minibatch SGD and apply the Adam solver {cite}`kingma2014adam` with learning rate $10^{-3}$; our [TensorFlow implementation is publicly available](https://github.com/predictive-analytics-lab/Data-Domain-Fairness).

#### Benchmarking and interpretability 
We enforce equality of opportunity as the fairness criterion, and we consider attractiveness as the positive label. 
Attractiveness is what could give someone a job opportunity or an advantaged outcome as defined in {cite}`HarPriSre16`. 
To test the hypothesis that we have learned a fairer image representation, we compare the performance and fairness of a standard SVM classifier trained using original images and the translated fair images. We use activation in the pool\_5 layer of the VGG19 network as features for training and evaluating the classifier[^3].

We report the quantitative results of this experiment in Table \ref{tab:results_celeba} (first and third rows) and the qualitative evaluations of image-to-image translations in Figure \ref{fig:faces}.  
From the Table \ref{tab:results_celeba} it is clear that the classifier trained on fair/translated images $\xtilde$ has improved over the classifier trained on the original images $\x$ in terms of equality of opportunity (reduction from $33.8$ to $23.8$) while maintaining the prediction accuracy ($79.4$ comparing to $80.6$). 
Looking at the TPR values across protected features (females and males), we can see that the male TPR value has increased, but it has an opposite effect for females. 
In the CelebA dataset, the proportion of attractive to unattractive males is around $30\%$ to $70\%$, and it is opposite for females; male group is therefore the minority group in this problem. 
Our method achieves better equality of opportunity measure than the baseline by increasing the minority group TPR value while decreasing the majority group TPR value.
To understand the balancing mechanism of TPR values (**Q3**), we visualize a subset of test male images that have been classified correctly as attractive after transformation (those examples were misclassified in the original domain) in Figure \ref{fig:faces}. 

\begin{figure}[t]
\begin{tabular}{ll}
\begin{tabular}{l}
\hspace{-0.4cm}input
\end{tabular}
\begin{tabular}{l}
\hspace{-0.32cm}
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/028288.jpg}
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/034362.jpg}
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/112024.jpg}
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/012307.jpg} 
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/013080.jpg} 
\end{tabular}\\
\begin{tabular}{l}
\hspace{-0.4cm}output 
\end{tabular}
\begin{tabular}{l}
\hspace{-0.5cm}
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/xtilde-028288.jpg}
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/xtilde-034362.jpg}
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/xtilde-112024.jpg}
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/xtilde-012307.jpg}
\includegraphics[width=0.08\textwidth]{figures/celeba_stargan/xtilde-013080.jpg}
\end{tabular}
\end{tabular}
\caption{Results of our approach (image-to-image translation via attributes). Given $N$ i.i.d. samples $\{(\x^n,y^n)\}_{n=1}^N$, our method transforms them into a new fair dataset $\{(\tilde{\x}^n,y^n)\}_{n=1}^{N}$ where $(\tilde{\x}^n,y^n)$ is the fair version of $(\x^n,y^n)$. 
The synthesized images are produced by the StarGAN model {cite}`ChoChoKimHaetal18` conditioned on the original images and their fair attribute representation. }
\label{fig:faces_attributes}
\end{figure}
\begin{figure}[t!b]
\centering
   \includegraphics[width=0.23\textwidth]{figures/Sattigeri_male_attractive_images/male_attractive_eopp/Male_1_Attractive_0_debias_1.png}
   \includegraphics[width=0.23\textwidth]{figures/Sattigeri_male_attractive_images/male_attractive_eopp/Male_1_Attractive_1_debias_1.png}
   \caption{Results of Fainess GAN {cite}`SatHofCheVar18` (Fig.2) of non-attractive (left) and attractive (right) males after pre-processing. 
   Given $N$ i.i.d. samples $\{(\x^n,y^n)\}_{n=1}^N$, Fainess GAN transforms them into a new fair dataset $\{(\tilde{\x}^n,\tilde{y}^n)\}_{n=1}^{N'}$ where $N'\neq N$ and $(\tilde{\x}^n,\tilde{y}^n)$ has no correspondence to $(\x^n,y^n)$.}
\label{fig:fairgan}
\end{figure}
\begin{figure}[t!b]
\centering
\includegraphics[width=0.38\textwidth]{figures/celeba_res/features.pdf}
\caption{Top 10 semantic attribute features that have been changed in $647$ males; those males were _incorrectly_ predicted as not attractive, but are now correctly predicted as attractive. $641$ and $639$ males out of $647$ are now with `Heavy_Makeup` and `Wearing_Lipstick` attributes, respectively, and $215$ out of $647$ males are now _without_ a `5_o_Clock_Shadow` attribute.}  
\label{fig:attributes}
\end{figure}
We observe a consistent localized area in face, specifically _lips_ and _eyes_ regions. 
The CelebA dataset has a large diversity in visual appearance of females and males (hair style, hair color) and their ethnic groups, so more localized facial areas have to be discovered to equalize TPR values across groups. 
Lips are very often coloured in female (the majority group) celebrity faces, hence our method, to increase the minority group TPR value, colorizes the lip regions of the minority group (males). 
Interestingly, female faces without prominent lipstick often got this transformation as well, prompting the decrease in the majority group TPR value. 
Regarding eye regions, several studies (e.g. {cite}`brown1993` and references therein) have shown their importance in gender identification. 
Also, a heavy makeup that is often applied to female celebrity eyes can also support our visualization in Figure \ref{fig:faces}.

The image-to-image translation using transformer network learns to produce coarse-grained changes, i.e. masking/colorizing face regions. 
This is expected as we learn a highly unconstrained mapping from source to target domain, in which the target data is unavailable. 
To enable fine-grained changes and semantic transformation of the images, we now explore semantic attributes; attributes are well-established interpretable mid-level representations for images. 
We show how an attribute-to-attribute translation provides an alternative way in analysing and performing an image-to-image translation. 

#### Attribute-to-attribute translation
Images in the CelebA dataset come with $40$ dimensional binary attribute annotations. We use all but two attributes (_gender_ and _attractiveness_) as semantic attribute representation of images. 
We then perform attribute-to-attribute translation with the transformer network and consider the same attractive versus not attractive task and gender protected characteristic as with the image data. 
We report the results of this experiment in Table \ref{tab:results_celeba} (second and forth rows correspond to the domain of attributes).
First, we observe that the predictive performance of the classifier trained on attribute representation is only slightly lower than the performance of the classifier trained on the image data ($79.1$ versus $80.6$), which enables sensible comparison of the results in these two settings. 
Second, we observe better gain in equality of opportunity when using the transformed attribute representation comparing to transformed images ($12.4$ is the best Eq. Opp. result in this experiment). 
This comes at the cost of a drop in accuracy performance. 
The TPR rates for both groups are higher when using translated attribute representation than when using translated image representation (third row versus fourth row). 
The largest improvement of the TPR is observed in the group of males (from $50.9$ in the original attribute to $74.8$ in the translated attribute space). 
Further analysis of changes in attribute representation reveals that equality of opportunity is achieved by putting _lipstick_ and _heavy-makeup_ to the male group (Figure \ref{fig:attributes}). 
These top 2 features have been mostly changed in the group of _males_. 
Very few changes happened in the group of females. 
This is encouraging as we have just arrived at the same conclusion (Figures \ref{fig:faces} and \ref{fig:attributes}), be it using images or using semantic attributes (**Q4**).

#### Image-to-image translation via attributes
Given the remarkable progress that has been made in the field towards image synthesis with the conditional GAN models, we attempt to synthesize images with respect to the attribute description.  
Specifically, we use the StarGAN model {cite}`ChoChoKimHaetal18`, the state-of-the-art model for image synthesis with multi-attribute transformation, to synthesize images with our learned fair attribute representation. 
For this, we pre-train the StarGAN model to perform image transformations with $38$ binary attributes (excluding gender and attractive attributes) using training data. 
We then translate all images in CelebA with respect to their fair attribute representation.  
We evaluate the performance of this approach and report the results in Table \ref{tab:results_celeba} (last row). 
We also include the qualitative evaluations of image-to-image translations via attributes in Figure \ref{fig:faces_attributes}. 
These visualizations essentially generalize counterfactual explanations in the sense of {cite}`WacMitRus18` to the image domain. 
We have just shown the "closest synthesized world", i.e. the smallest change to the world that can be made to obtain a desirable outcome.
Overall, the classifier trained using this fair representation shows similar Eq. Opp. performance and comparable accuracy to the classifier trained on representation learned with the transformer network. 
However, the TPR rates for both protected groups are higher (last row versus third row), especially in the group of males, when using this representation. 

#### Pre-processing approaches
The aim of the pre-processing approaches such as {cite}`SatHofCheVar18, calmon2017optimized` is to transform the given dataset of $N$ i.i.d. samples $\{(\x^n,y^n)\}_{n=1}^N$ into a _new_ fair dataset $\{(\tilde{\x}^n,\tilde{y}^n)\}_{n=1}^{N'}$. 
It is important to note that $N'$ is not necessarily equal to $N$, and therefore $(\tilde{\x}^n,\bar{y}^n)$ has no correspondence to $(\x^n,y^n)$. 
{cite}`calmon2017optimized` has proposed this approach for tabular (discrete) data, while {cite}`SatHofCheVar18` has explored image data. 
Here, we offer a unified framework for tabular (continuous and discrete) and image data that transforms the given dataset $\{(\x^n,y^n)\}_{n=1}^N$ into a new fair dataset $\{(\tilde{\x}^n,y^n)\}_{n=1}^{N}$ where $(\tilde{\x}^n,y^n)$ is the fair version of $(\x^n,y^n)$. 
*What is the advantage of creating a fair representation per sample (our method) rather than on the whole dataset at once {cite}`SatHofCheVar18, calmon2017optimized`?*
The first can be used to provide an _individual_-level explanation of fair systems, while the latter can only be used to provide a _system_-level explanation.
For comparison, we include here a snapshot of results presented in {cite}`SatHofCheVar18` using the CelebA dataset in Figure \ref{fig:fairgan}. 
The figure shows eigenfaces/eigensketches with _the mean image_ of the new fair dataset $\{(\tilde{\x}^n)\}_{n=1}^{N'}$ (in the center) of the $3\times 3$ grid.
No per sample visualisation $(\tilde{\x}^n)$ was provided.
Left/right/top/bottom images in Fig. \ref{fig:fairgan} show variations along the first/second principal components. 
In contrast, Figure \ref{fig:faces_attributes} shows a per sample visualisation $(\tilde{\x}^n)$ using our proposed method.

### The Diversity in Faces dataset
We extract and align face crops from the images and use 128x128 facial images as the inputs. 
Our preliminary experiment has similar setup to the image-to-image translation on the CelebA dataset except that the prediction task has seven age groups to be classified.  
As the fairness criterion we enforce equality of opportunity considering the middle age group (31-45) to be desirable (as the positive label when conditioning). 
As before, to test the hypothesis that we have learned a fairer image representation, we compare the performance and fairness of the SVM classifier trained using original images and the translated fair images (with features as activations in the pool\_5 layer of the VGG19 network). 
We achieve $52.85$ as the overall classification accuracy over seven age groups when using original image features and an increased $60.26$ accuracy when using translated images. 
The equality of opportunity improved from $27.21$ using original image representation to $9.85$ using fair image representation. 
Similarly to the CelebA dataset, the image-to-image translation using transformer network learns to produce coarse-grained changes, i.e. masking/colorizing nose regions (as opposed to lips and eyes regions on CelebA). 
These preliminary results are encouraging and further analysis will be addressed as a future extension.  

[^1]: Examples of fairness criteria are equality of true positive rates (TPR), also called equality of opportunity {cite}`HarPriSre16,ZafValRodGum17b`, between males and females.
[^2]: With binary labels, it is assumed that positive label is a desirable/advantaged outcome, e.g. expected to perform well at the job.
[^3]: We deliberately evaluate the performance (accuracy and fairness) using an auxiliary classifier instead of using the predictor of the transformer network. Since the emphasis of this work is on representation learning, we should not prescribe what classifier the user chooses on top of learned representation.

---
```{bibliography}
:filter: docname in docnames
```
