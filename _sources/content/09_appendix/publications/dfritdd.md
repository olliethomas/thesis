# Discovering Fair Representations in the Data Domain

Authors: N.Quadrianto, V. Sharmanska, O. Thomas

---

Published at IEEE Conference on Computer Vision and Pattern Recognition ([^CVPR]) 2019

```bibtex
@InProceedings{QuaShaTho19,
  author    = {Quadrianto, Novi and 
               Sharmanska, Viktoriia and 
               Thomas, Oliver},
  title     = {Discovering Fair Representations in the Data Domain},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2019}
}
```


[landing page](https://openaccess.thecvf.com/content_CVPR_2019/html/Quadrianto_Discovering_Fair_Representations_in_the_Data_Domain_CVPR_2019_paper.html)
| [pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Quadrianto_Discovering_Fair_Representations_in_the_Data_Domain_CVPR_2019_paper.pdf)
| [code](https://github.com/predictive-analytics-lab/Data-Domain-Fairness)

**Abstract**: Interpretability and fairness are critical in computer vision and machine learning applications, in particular when dealing with human outcomes, e.g. inviting or not inviting for a job interview based on application materials that may include photographs. 
One promising direction to achieve fairness is by learning data representations that remove the semantics of protected characteristics, and are therefore able to mitigate unfair outcomes. 
All available models however learn latent embeddings which comes at the cost of being uninterpretable. 
We propose to cast this problem as data-to-data translation, i.e. learning a mapping from an input domain to a fair target domain, where a fairness definition is being enforced. 
Here the data domain can be images, or any tabular data representation. 
This task would be straightforward if we had fair target data available, but this is not the case. 
To overcome this, we learn a highly unconstrained mapping by exploiting statistics of residuals – the difference between input data and its translated version – and the protected characteristics. 
When applied to the CelebA dataset of face images with gender attribute as the protected characteristic, our model enforces equality of opportunity by adjusting the eyes and lips regions. 
Intriguingly, on the same dataset we arrive at similar conclusions when using semantic attribute representations of images for translation. 
On face images of the recent DiF dataset, with the same gender attribute, our method adjusts nose regions. 
In the Adult income dataset, also with protected gender attribute, our model achieves equality of opportunity by, among others, obfuscating the wife and husband relationship. 
Analyzing those systematic changes will allow us to scrutinize the interplay of fairness criterion, chosen protected characteristics, and prediction performance.

[^CVPR]: ERA rating A (best), QUALIS rating A1 (best), CORE rank A* (best), 25.2% acceptance rate (2019)
