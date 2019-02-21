# DailyReadPaper
Today I will summarize the recent Normalization Methods for Training Deep Neural Networks: Theory and Practice

## Decorrelated batch normalization
1. CVPR 2018
2. Huang, Lei and Yang, Dawei and Lang, Bo and Deng, Jia
3. 190208(1)Huang_Decorrelated_Batch_Normalization_CVPR_2018_paper.pdf

- Batch Normalization (BN) is capable of accelerating the
training of deep models by centering and scaling activations
within mini-batches. In this work, we propose Decorrelated Batch Normalization (DBN), which not just centers
and scales activations but whitens them.

    ![reid](Pictures/1.png)
    
- We explore multiple
whitening techniques, and find that PCA whitening causes a
problem we call stochastic axis swapping, which is detrimental to learning. We show that ZCA whitening does not suffer
from this problem, permitting successful learning. DBN retains the desirable qualities of BN and further improves BN’s
optimization efficiency and generalization ability. We design
comprehensive experiments to show that DBN can improve
the performance of BN on multilayer perceptrons and convolutional neural networks. Furthermore, we consistently
improve the accuracy of residual networks on CIFAR-10,
CIFAR-100, and ImageNet.


>@inproceedings{huang2018decorrelated,
  title={Decorrelated batch normalization},
  author={Huang, Lei and Yang, Dawei and Lang, Bo and Deng, Jia},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={791--800},
  year={2018}
}