# Front Matter

## Title: Fair Representations in the Data Domain

```{image} https://i.imgur.com/F2Y91nd.jpg
:alt: simple vs complex
:class: bg-primary mb-1
:width: 600px
:align: center
```

## Abstract

Algorithmic fairness is a multi-faceted topic which is of significant consequence to a diverse range of people.
The issue that this thesis investigates is a fairness-specific instance of a yet even broader concern -- that data can be biased due to spurious correlations.
Machine learning models trained on such data learn to exploit these spurious correlations that do not hold in the test distribution.
When spurious correlations are found with respect to protected demographic attributes, trained models could be biased towards certain subgroups or populations.
A promising approach to counteract biased data is by producing a fair representation as a pre-processing step.
The main drawback, however, of existing fair representation learning approaches is that the data often becomes obscured when projected into an uninterpretable latent space, making intuitive assessment difficult.
Noticing that the domain the data resides in is often interpretable, with the structure providing richer information that is easier to understand on a per sample basis, I develop fair representations in the data domain.
These convey additional per-sample information that can be easily shared and explained to system designers and stakeholders.
This thesis investigates three aspects of fair representations in the data domain.
Firstly, I demonstrate a novel application of fair representations to generate counterfactual samples in the data domain.
The aim of this application is to promote positive actions to address discrimination in an already existing system; 
Secondly, I develop a method to produce fair representations in the data domain based on statistical dependence principles; 
Lastly, I take this approach further, introducing two further methods to achieve fair representations in the data domain based on adversarial learning.

