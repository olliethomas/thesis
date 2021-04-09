# Activities to date

## Previous Years

### 2017-2018: Part-Time

#### Work

- Looked into evolutionary algorithms for finding models on the Pareto frontier of the utility / fairness trade­off.
    - This was an initial task when I first joined the group. It provided a good chance to experiment with the topic, though isn't something I've pursued.
- Worked on adversarial approach to fairness with Alexey Umnov based on Wasserstein distance.
    - I began working on this in collaboration with Alexey from the research group at HSE. We made some progress, but 
    we weren't able to outperform state of the art methods.    
- Implemented changes to AlgoFairness framework so that it meets PAL needs.
    - AlgoFairness is a framework provided by the group at Haverford College, PA. 
    They have a research group that is also interested in algorithmic fairness.
    We tried to use the tools that they provided, but found them too limiting for the research we wanted to do.
    After spending some time repurposing their tools to suit our reseaech needs we decided to abandon this package and create our
    own algorithmic fairness library, EthicML. 
- Worked with colleagues to submit a paper to NeurIPS[^neurips2018] inspired by style-transfer.
    - The idea was to change an image to a fair version of itself by applying a "fairness mask". 
    - Whilst the paper was rejected from this conference, we worked on the feedback and re-submitted to CVPR. (See 2018-2019)
- Began working on EthicML python package.

[^neurips2018]: ERA rating A (best), QUALIS rating A1 (best), 20.8% acceptance rate (2018).

#### Activity within the group

- Regular presentations (and attendance) within PAL reading group. Talks given:­
  - Iterative Machine Teaching.
  - Fair and Transferable Representations.
  - The Delayed Impact of Fair Machine Learning.
  - Path­Specific Counterfactual Fairness.
- Gave 2 lectures and 2 seminars for the MSc course "Topics in Computing" on the topic of Ethical ML.

### 2018-2019: Full-Time

#### Work

- We extended the previous NeurIPS submission and submitted to CVPR[^cvpr2019] [our paper on discovering fair representations in the data domain](../09_appendix/publications/dfritdd.md).
    - We changed the focus to be less on styling and more on the problem of training with an unknown target. 
    The problem identified of finding a "fair" version of an image is not something that can be obviously solved with a typical GAN approach.
    - The paper was **accepted** and published at the main conference.
    - This will form part of chapter 1 of my thesis.  
- IJCAI[^ijcai2019] [submission on using generative models to balance a dataset with regard to
  sensitive groups and their outcomes](../../../../Library/Application%20Support/JetBrains/PyCharm2020.3/scratches/imagined.md).
  - Having seen other work focusing on the input data and ignoring the outcomes as a source of bias, we submitted
  a paper to IJCAI which aimed to disentangle the effect of a sensitive attribute from both the input space and the decision space.
  - The paper was rejected with feedback that the paper was poorly structured, which was fair. 
  - This paper was worked on and submitted to AAAI (see 2019-2020). 
- NeurIPS[^neurips2019] [submission on censoring a representation using an invertible network](../09_appendix/publications/nosinn.md).
    - Having spoken with colleagues about a new method they were keen to experiment with, Invertible Neural Networks (INNs),
    I spoke with them about my recent work and how I was approaching the problem of disentangling a sensitive attribute
    in latent space, and casting back into the input space, modifying the sensitive partition.
    We agreed that this would be a suitable problem to use an INN.
    - We submitted this work to NeurIPS.
    - It was rejected with the feedback that the results weren't convincing.
    - We worked on this and resubmitted to CVPR (see 2019-2020).
- Written [EthicML](https://pypi.org/project/EthicML/) python library for measuring fairness (`pip install EthicML`).
    - Throughout the year I worked more on EthicML, encouraging others in the group who are also working on Algorithmic Fairness
    to also use the library which they not only did, but also considerably contributed to.
    - This is relevant to my thesis as it includes datasets, baseline models and metrics related to algorithmic fairness. 
    
[^cvpr2019]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A* (best), 25.2% acceptance rate (2019).
[^ijcai2019]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A* (best), 17.88% acceptance rate (2019).
[^neurips2019]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A* (best), 21.6% acceptance rate (2019).

#### Activity within the group
- Gave 2 lectures on Fairness in Machine Learning at the Discus Machine Learning Summer School.

## This Year

### 2019-2020: Full-Time

Since the last annual review, I worked on 3 papers

1. _[Imagined Examples](../../../../Library/Application%20Support/JetBrains/PyCharm2020.3/scratches/imagined.md):_ This is a re-work of the IJCAI submission from the previous year.
We use a Variational Autoencoder model to produce a model that not 
only considers the features, but also the reported outcomes as a source of bias. 
We use this model to generate likely counterfactual samples but without the need for a causal model.
We focussed more on explaining that the model can not only identify, but remedy 2 sources of bias, sample bias and proxy labels.
We also paid more attention to the overlap with the fairness is causality literature. 
    - We submitted to AAAI[^aaai2020], but the paper was rejected due to the results being difficult to interpret. 
    - This is now continuing work.
    - This will form chapter 2 of my thesis.

2. _[Null-sampling for Invariant and Interpretable Representations](../09_appendix/publications/nosinn.md):_ We use the bijective 
property of invertible neural networks in conjunction with an adversarial network 
to _disentangle_ a sensitive attribute's effect on other features. 
We then generate samples that are neutral with respect to a sensitive attribute, using these neutral images as our 
representation of the data we can investigate how the model has made the data "fairer".
This work was previously rejected from NeurIPS.
    - We submitted the paper to CVPR[^cvpr2020], but the reviews were that whilst the approach was novel, the experiments were a little weak and was rejected.
    - We improved the clarity of the results and re-submitted to ECCV[^eccv2020], where the paper has been accepted to be published at the main conference.
    - This will form part of chapter 1 of my thesis.
 
3. _[Invisible Demographics](../09_appendix/publications/invisible.md):_ Training data doesn't always reflect the deployment setting, 
with some demographic groups either mis-represented, or simply missing. 
We use unlabelled, but representative, data to guide our learned representations so that demographic groups aren't missed.
    - This work was submitted to NeurIPS and is currently under review.

[^aaai2020]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A* (best), 20.6% acceptance rate (2020).
[^cvpr2020]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A* (best), 22.1% acceptance rate (2020).
[^eccv2020]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A, 27% acceptance rate (2020).

### Implementations
In addition I managed the Surgo reseach project for 7 months to produce the causal modelling tool [Intervene](https://github.com/predictive-analytics-lab/Intervene).

#### Intervene
As part of the Surgo research project we created the tool Intervene. 
This tool has 3 core functionalities:
1. Creating data
2. Fitting a Casual Model to data.
3. Evaluating the performance of the learned model.


##### Causal Modelling
Whilst the topic of causality is a research topic in it's own rite, there are clearly parallels between this area and 
the broad category of my research area - algorithmic fairness, accountability and transparency.

If we were able to accurately model the causal relationships within the data, then we could ask fundamental questions,
interrogating the underlying frameworks that introduce bias into our observations.
However, **accurate** causal modelling is not a solved problem.
My hope when beginning this research project was to investigate the effect of the learned causal relationship between
raw data and fair representations. 
The motivating question being, "does a fair representation change the observed causal relationships in data?".
This is a challenging problem as most fair representations occupy some latent space that is different to that where the
data lies.
However, my previous research has been specific to learning a fair representation in the data domain, making such a 
comparison possible. 
Investigating this will (possibly) form chapter 3 of of my thesis.


### Other works
- Gave a [presentation](https://predictive-analytics-lab.github.io/presentations/toronto2019.html#/) on fairness in machine learning 
at [Conference on Data Science and Optimization](http://www.fields.utoronto.ca/talks/Transparency-fairness) in Toronto.
- Presented at the PAL reading group 5 times.

#### Training Courses Attended
- Preparing for your Viva. Organised by the Doctoral School.
- How to Edit Your Own Writing. Organised by the Doctoral School.
- Master Your Workload. Organised by the Doctoral School.

## 2020 onwards: Continuing Work

- Expand the CVPR '19 paper to a journal entry (see [thesis structure section](./structure.md#discovering-fair-representations-in-the-data-domain)).
- Imagined examples paper.
- Depending on choice of chapter 3 (see [thesis structure](./structure.md#chapter-3)).
    - Investigating the use of Active Learning to model changing demographics.
    - Characterising the relationship between fair learnt representations and counterfactual samples.
- Writing Introduction and Background for thesis. 

## Plan

### Last year's activity plan

| Activity | Q3 '19 | Q4 '19 | Q1 '20 | Q2 '20 | Q3 '20 | Q4 '20 | Q1 '21 | Q2 '21 |
| :------: | :--------: | :-----------: | :---------: | :----------: | :----------: | :---------: | :----------: | :-------: |
| Imagined Examples re-submission | X |  |  |  |  |  |  |  |
| Intervene |  | X | X | X |  |  |  |  |
| Causality & Fairness |  |  |  | X | X | X |  |  |
| Thesis |  |  |  |  |  | X | X | X |


### This year's update of activity plan

| Activity | August '20 | September '20 | October '20 | November '20 | December '20 | January '21 | February '21 | March '22 | April '21 | May '21 | June '21 | July '21 |
| :------: | :--------: | :-----------: | :---------: | :----------: | :----------: | :---------: | :----------: | :-------: | :-------: | :-----: | :------: | :------: |
| Journal submission based on Discovering Fair Representations CVPR paper | X | X | X |  |  |  |  |  |  |  |  |  |
| Imagined Examples re-submission |  | X | X |  |  |  |  |  |  |  |  |  |
| Chapter 3 work |  |  |  | X | X | X |  |  |  |  |  |  |
| Thesis Writing |  |  |  |  |  |  |  | X | X | X | X |  |

