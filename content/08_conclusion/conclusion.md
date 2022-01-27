# Chapter 7: Conclusion

A concern with following any purely statistical route in knowing in
advance some attribute that you wish to be fair to. If we consider the
UCI Adult Dataset (a popular dataset for benchmarking fairness
algorithms with features such as Gender, Race and Nationality) and you
wish to be fair with regard to _religion_, a feature which is not been
captured in this dataset (though could be inferred using prior knowledge
and the nationality feature), you will not be able to claim fairness
with regard to that non-captured attribute, despite Freedom of
Religion[^9] being enshrined in most modern democratic countries. There
has been an attempt to address this problem in the paper Proxy
Fairness {cite}`GupCotFarWan18` which finds potential proxy groups
that could potentially be discriminated against, though this is a first
foray into this particularly difficult problem.

Many classifiers are focused on making predictions without needing
sensitive attributes to be captured. Whilst this may help achieve
short-term goals, none of the methods discussed would be able to work if
that information were not captured in the first place. It may seem
counter-intuitive, but to ensure the long-term goals of ethical machine
learning, of fairer, more interpretable, more transparent and more
accountable algorithms, we need sensitive attributes to be recorded. We
may not use these features within our models, but if not captured, we
stand little chance of calming ethical concerns about the application of
our field to subjective areas.

---
```{bibliography}
:filter: docname in docnames
```

[^1]: Certainly from the ethical perspective, although interpretability of machine learning models is a research area in it's own right.
[^2]: And alternatively, we may be able to interpret some stages of the decision making process but the process to obtain these may be opaque to us.
[^3]: For more on this paper, see {ref}`causal`.
[^4]: Throughout this section we consider a binary sensitive attribute, though all definitions can be extended to accommodate a sensitive attribute with $n$ categories.
[^5]: Which due to contractual reasons they are unable to release.
[^6]: In this case, with regard to demographic parity.
[^7]: The CelebA, Soccer Player and Quickdraw datasets were used.
[^8]: Given the arbitrariness of many 'fairness' rules in the U.S. that have led to claims of disparate treatment (i.e. Ricci vs DeStefano or Texas House Bill 588) this objective may have a large impact.
[^9]: And consequently the expectation to be free of discrimination based on that religion.
