# 


>**This is the official implementation of the paper with the title “A Human Activity Recognition Method Based on Lightweight Feature Extraction Combined with Pruned and Quantized CNN for Wearable Devices” by Myung-Kyu Yi and Seong-Oun Hwang.**

## Paper Overview

**Abstract**: Human Activity Recognition (HAR) has garnered attention as a significant technology that can be enhancing the quality of human life. However, existing HAR works still face great challenges such as a shortage of labeled data and the difficulty of rebuilding a deep learning model whenever the application environment (e.g., user or sensor position) changes. To address these challenges, we propose a new data-centric approach for HAR by using a Semi-supervised Generative Adversarial Network (SGAN). To improve the accuracy of HAR, we propose a data supplement strategy that systematically improves data quality, rather than the model, by using data refinement and data-driven feature extraction techniques. The proposed HAR method applies simple SGAN to achieve considerably high accuracy with only a small fraction of the labeled data. Therefore, the proposed HAR method can reduce overhead from data labeling, which is a labor-intensive and time-consuming process for many HAR tasks. Moreover, the data-centric HAR method is robust even in scenarios when there is a change in person/sensor location. Experimental results show that our method improves accuracy by as much as 3% over state-of-the-art semi-supervised HAR methods with only 3% of the data being labeled, leading to comparable accuracy to state-of-the-art HAR methods based on supervised learning.

---
## Codebase Overview
- We implement our proposed HAR into the same framework. We note that:
  - Running ''Data-driven_Feature_Extraction'' for our proposed data-feaute extaction technique.
  - Running ''Data-refinement'' for our proposed data refinement technique.
  - Running ''_main_baselines.py_'' for baseline SGAN model. 
  
## Citing This Repository

Please cite our paper if you find this repo useful in your work:

```
@InProceedings{pmlr-v202-li23s,
  title = 	 {Revisiting Weighted Aggregation in Federated Learning with Neural Networks},
  author =       {Li, Zexi and Lin, Tao and Shang, Xinyi and Wu, Chao},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {19767--19788},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/li23s/li23s.pdf},
  url = 	 {https://proceedings.mlr.press/v202/li23s.html},
  abstract = 	 {In federated learning (FL), weighted aggregation of local models is conducted to generate a global model, and the aggregation weights are normalized (the sum of weights is 1) and proportional to the local data sizes. In this paper, we revisit the weighted aggregation process and gain new insights into the training dynamics of FL. First, we find that the sum of weights can be smaller than 1, causing global weight shrinking effect (analogous to weight decay) and improving generalization. We explore how the optimal shrinking factor is affected by clients’ data heterogeneity and local epochs. Second, we dive into the relative aggregation weights among clients to depict the clients’ importance. We develop client coherence to study the learning dynamics and find a critical point that exists. Before entering the critical point, more coherent clients play more essential roles in generalization. Based on the above insights, we propose an effective method for Federated Learning with Learnable Aggregation Weights, named as FedLAW. Extensive experiments verify that our method can improve the generalization of the global model by a large margin on different datasets and models.}
}
```

## Contact

Please feel free to contact via email (<zexi.li@zju.edu.cn>) if you have further questions.
