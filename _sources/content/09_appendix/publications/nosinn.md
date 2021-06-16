# Null-Sampling for Invariant and Interpretable Representations

Authors: T. Kehrenberg, M. Bartlett, O. Thomas, N. Quadrianto

---

Published at European Conference of Computer Vision ([^ECCV]) 2020.

```bibtex
@InProceedings{KehBarThoQua20,
  author    = {Kehrenberg, Thomas and 
               Bartlett, Myles and 
               Thomas, Oliver and 
               Quadrianto, Novi},
  editor    = {Vedaldi, Andrea and Bischof, Horst and Brox, Thomas and Frahm, Jan-Michael},
  title     = {Null-Sampling for Interpretable and Fair Representations},
  booktitle = {Computer Vision -- ECCV 2020},
  year      = {2020},
  publisher = {Springer International Publishing},
  address   = {Cham},
  pages     = {565--580},
  isbn      = {978-3-030-58574-7}
}
```

[landing page](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5488_ECCV_2020_paper.php)
| [pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710562.pdf)
| [publisher's page](https://link.springer.com/chapter/10.1007/978-3-030-58574-7_34)
| [code](https://github.com/predictive-analytics-lab/nifr)


**Abstract**: We propose to learn invariant representations, in the data domain, to achieve interpretability in algorithmic fairness. 
Invariance implies a selectivity for high level, relevant correlations w.r.t. class label annotations, and a robustness to irrelevant correlations with protected characteristics such as race or gender. 
We introduce a non-trivial setup in which the training set exhibits a strong bias such that class label annotations are irrelevant and spurious correlations cannot be distinguished. 
To address this problem, we introduce an adversarially trained model with a null-sampling procedure to produce invariant representations in the data domain. 
To enable disentanglement, a partially-labelled representative set is used. 
By placing the representations into the data domain, the changes made by the model are easily examinable by human auditors. 
We show the effectiveness of our method on both image and tabular datasets: Coloured MNIST, the CelebA and the Adult dataset."


[^ECCV]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A, 27% acceptance rate (2020)
