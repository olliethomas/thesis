# Data Domain Fairness

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


[^1]: Examples of fairness criteria are equality of true positive rates (TPR), also called equality of opportunity {cite}`HarPriSre16,ZafValRodGum17b`, between males and females.


---
```{bibliography}
:filter: docname in docnames
```
