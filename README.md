# 


>**This is the official implementation of the paper with the title “A Data-driven Feature Extraction Method Based on Data Supplement for Human Activity Recognition”by Myung-Kyu Yi and Seong-Oun Hwang.**

## Paper Overview

**Abstract**: Human Activity Recognition (HAR) has garnered attention as a significant technology that can enhance the quality of human life. However, existing HAR works still face great challenges such as a shortage of labeled data and the difficulty of rebuilding a deep learning model whenever the application environment (e.g., user or sensor position) changes. To address these challenges, we propose a new data-centric approach for HAR by using a Semi-supervised Generative Adversarial Network (SGAN). To improve the accuracy of HAR, we propose a data supplement strategy that systematically improves data quality, rather than the model, by using data refinement and data-driven feature extraction techniques. The proposed HAR method applies simple SGAN to achieve considerably high accuracy with only a small fraction of the labeled data. Therefore, the proposed HAR method can reduce overhead from data labeling, which is a labor-intensive and time-consuming process for many HAR tasks. Moreover, the data-centric HAR method is robust even in scenarios when there is a change in person/sensor location. Experimental results show that our method improves accuracy by as much as 3% over state-of-the-art semi-supervised HAR methods with only 3% of the data being labeled, leading to comparable accuracy to state-of-the-art HAR methods based on supervised learning.

---
## Dataset
- PAMAP2 dataset is available at https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
- DSADS dataset is available at https://archive.ics.uci.edu/dataset/256/daily+and+sports+activitie
- MobiAct dataset is available at https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2

## Codebase Overview
- We note that:
  - <Data-driven_Feature_Extraction.ipynb> for our proposed data-feaute extaction technique.
  - <Data-refinement.ipynb> for our proposed data refinement technique.
  - <SGAN.ipynb> for baseline SGAN model

Framework uses Tensorflow 2+, tensorflow_addons, numpy, pandas, matplotlib, scikit-learn.  
  
## Citing This Repository

If our project is helpful for your research, please consider citing :

```

@inJournal{XXX,
  title={A Data-driven Feature Extraction Method Based on Data Supplement for Human Activity Recognition},
  author={Myung-Kyu Yi and Seong-Oun Hwang},
  booktitle={IEEE XXX},
  year={2023}
}

```

## Contact

Please feel free to contact via email (<kainos@gachon.ac.kr>) if you have further questions.
