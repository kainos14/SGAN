# 


>**This is the official implementation of the paper with the title “A Human Activity Recognition Method Based on Lightweight Feature Extraction Combined with Pruned and Quantized CNN for Wearable Devices”by Myung-Kyu Yi and Seong-Oun Hwang.**

## Paper Overview

**Abstract**: Human Activity Recognition (HAR) has garnered attention as a significant technology that can be enhancing the quality of human life. However, existing HAR works still face great challenges such as a shortage of labeled data and the difficulty of rebuilding a deep learning model whenever the application environment (e.g., user or sensor position) changes. To address these challenges, we propose a new data-centric approach for HAR by using a Semi-supervised Generative Adversarial Network (SGAN). To improve the accuracy of HAR, we propose a data supplement strategy that systematically improves data quality, rather than the model, by using data refinement and data-driven feature extraction techniques. The proposed HAR method applies simple SGAN to achieve considerably high accuracy with only a small fraction of the labeled data. Therefore, the proposed HAR method can reduce overhead from data labeling, which is a labor-intensive and time-consuming process for many HAR tasks. Moreover, the data-centric HAR method is robust even in scenarios when there is a change in person/sensor location. Experimental results show that our method improves accuracy by as much as 3% over state-of-the-art semi-supervised HAR methods with only 3% of the data being labeled, leading to comparable accuracy to state-of-the-art HAR methods based on supervised learning.

---
## Codebase Overview
- We note that:
  - <Data-driven_Feature_Extraction.ipynb> for our proposed data-feaute extaction technique.
  - <Data-refinement.ipynb> for our proposed data refinement technique.
  - <SGAN.ipynb> for baseline SGAN model. 
  
## Citing This Repository

If our project is helpful for your research, please consider citing :

```

@misc{https://doi.org/1,
  doi = {},
  url = {},
  author = {Myung-Kyu Yi and Seong-Oun Hwang},
  keywords = {human activity recognition, data-centric AI, wearable device, generative AI, semi-supervised learning},
  title = {A Human Activity Recognition Method Based on Lightweight Feature Extraction Combined with Pruned and Quantized CNN for Wearable Devices},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

## Contact

Please feel free to contact via email (<kainos@gachon.ac.kr>) if you have further questions.
