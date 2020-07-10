# Previous Years

```{admonition} WIP
:class: Tip
This content is work in progress.
```

## 2017-2018: Part-Time

### Work

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
    own algortithmic fairness library, EthicML. 
- Worked with Novi and Viktoriia to submit a paper to NeurIPS inspired by style-transfer.
    - The idea was to change an image to a fair version of itself by applying a "fairness mask". 
    - Whilst the paper was rejected from this conference, we worked on the feedback and re-submitted to CVPR. (See 2018-2019)
- Began working on EthicML python package.

### Activity within the group

- Regular presentations (and attendance) within PAL reading group. Talks given:­
  - Iterative Machine Teaching.
  - Fair and Transferable Representations.
  - The Delayed Impact of Fair Machine Learning.
  - Path­Specific Counterfactual Fairness.
- Gave 2 lectures and 2 seminars for the MSc course "Topics in Computing" on the topic of Ethical ML.

## 2018-2019: Full-Time

### Work

- We extended the previous NeurIPS submission and submitted to CVPR [our paper on discovering fair representations in the data domain](../09_appendix/dfritdd.md).
    - We changed the focus to be less on styling and more on the problem of training with an unknown target. 
    The problem identified of finding a "fair" version of an image is not something that can be obviously solved with a typical GAN approach.
    The paper was accepted and published at the main conference.  
- [IJCAI submission on using generative models to balance a dataset with regard to
  sensitive groups and their outcomes](../09_appendix/imagined.md).
  - Having seen other work focusing on 
- [NeurIPS submission on censoring a representation using an invertible network](../09_appendix/nosinn.md).
- Written [EthicML](https://pypi.org/project/EthicML/) python library for measuring fairness (`pip install EthicML`).
- Taught several lectures on Fairness in Machine Learning.
