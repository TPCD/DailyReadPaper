# DailyReadPaper
Today I will summarize some resent Adversarial Training methods.

## Attention-Aware Deep Adversarial Hashing for Cross-Modal Retrieval
1. ECCV 2018
2. Zhang, Xi and Lai, Hanjiang and Feng, Jiashi
3. 190212(1)Xi_Zhang_Attention-aware_Deep_Adversarial_ECCV_2018_paper.pdf

- Due to the rapid growth of multi-modal data, hashing meth-
ods for cross-modal retrieval have received considerable attention. How-
ever, finding content similarities between different modalities of data is
still challenging due to an existing heterogeneity gap.

    ![reid](Pictures/Selection_356.png)

- To further address
this problem, we propose an adversarial hashing network with an atten-
tion mechanism to enhance the measurement of content similarities by
selectively focusing on the informative parts of multi-modal data. The
proposed new deep adversarial network consists of three building blocks:
1) the feature learning module to obtain the feature representations; 2)
the attention module to generate an attention mask, which is used to
divide the feature representations into the attended and unattended fea-
ture representations; and 3) the hashing module to learn hash functions
that preserve the similarities between different modalities. In our frame-
work, the attention and hashing modules are trained in an adversarial
way: the attention module attempts to make the hashing module un-
able to preserve the similarities of multi-modal data w.r.t. the unattend-
ed feature representations, while the hashing module aims to preserve
the similarities of multi-modal data w.r.t. the attended and unattend-
ed feature representations. Extensive evaluations on several benchmark
datasets demonstrate that the proposed method brings substantial im-
provements over other state-of-the-art cross-modal hashing methods.

>@inproceedings{zhang2018attention,
  title={Attention-Aware Deep Adversarial Hashing for Cross-Modal Retrieval},
  author={Zhang, Xi and Lai, Hanjiang and Feng, Jiashi},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={591--606},
  year={2018}
}


## Cosface: Large margin cosine loss for deep face recognition
1. CVPR 2018
2. Wang, Hao and Wang, Yitong and Zhou, Zheng and Ji, Xing and Gong, Dihong and Zhou, Jingchao and Li, Zhifeng and Liu, Wei
3. 190211(2)Wang_CosFace_Large_Margin_CVPR_2018_paper.pdf

- Face recognition has made extraordinary progress ow-
ing to the advancement of deep convolutional neural net-
works (CNNs). The central task of face recognition, in-
cluding face verification and identification, involves face
feature discrimination. However, the traditional softmax
loss of deep CNNs usually lacks the power of discrimina-
tion. To address this problem, recently several loss func-
tions such as center loss, large margin softmax loss, and
angular softmax loss have been proposed.

    ![reid](Pictures/Selection_348.png)

- All these improved losses share the same idea: maximizing inter-class
variance and minimizing intra-class variance. In this pa-
per, we propose a novel loss function, namely large mar-
gin cosine loss (LMCL), to realize this idea from a different
perspective. More specifically, we reformulate the softmax
loss as a cosine loss by L 2 normalizing both features and
weight vectors to remove radial variations, based on which
a cosine margin term is introduced to further maximize the
decision margin in the angular space.

    ![reid](Pictures/Selection_349.png)

- As a result, minimum intra-class variance and maximum inter-class variance are
achieved by virtue of normalization and cosine decision
margin maximization. We refer to our model trained with
LMCL as CosFace. Extensive experimental evaluations are
conducted on the most popular public-domain face recogni-
tion datasets such as MegaFace Challenge, Youtube Faces
(YTF) and Labeled Face in the Wild (LFW). We achieve the
state-of-the-art performance on these benchmarks, which
confirms the effectiveness of our proposed approach.

>@inproceedings{wang2018cosface,
  title={Cosface: Large margin cosine loss for deep face recognition},
  author={Wang, Hao and Wang, Yitong and Zhou, Zheng and Ji, Xing and Gong, Dihong and Zhou, Jingchao and Li, Zhifeng and Liu, Wei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5265--5274},
  year={2018}
}

## In Defense of the Triplet Loss for Visual Recognition
1. Arxiv 2019
2. Taha, Ahmed and Chen, Yi-Ting and Misu, Teruhisa and Davis, Larry
3. 1902011(3)In Defense of the Triplet Loss for Visual Recognition.pdf
- We employ triplet loss as a space embedding regular-
izer to boost classification performance. Standard archi-
tectures, like ResNet and DesneNet, are extended to sup-
port both losses with minimal hyper-parameter tuning. This
promotes generality while fine-tuning pretrained networks.
Triplet loss is a powerful surrogate for recently proposed
embedding regularizers.

    ![reid](Pictures/Selection_351.png)

- Yet, it is avoided for large batch-
size requirement and high computational cost. Through our
experiments, we re-assess these assumptions.
During inference, our network supports both classifica-
tion and embedding tasks without any computational over-
head.

    ![reid](Pictures/Selection_352.png)

- Quantitative evaluation highlights how our approach
compares favorably to the existing state of the art on multi-
ple fine-grained recognition datasets. Further evaluation
on an imbalanced video dataset achieves significant im-
provement (> 7%). Beyond boosting efficiency, triplet loss
brings retrieval and interpretability to classification mod-
els.
>@article{taha2019defense,
  title={In Defense of the Triplet Loss for Visual Recognition},
  author={Taha, Ahmed and Chen, Yi-Ting and Misu, Teruhisa and Davis, Larry},
  journal={arXiv preprint arXiv:1901.08616},
  year={2019}
}

## Support Vector Guided Softmax Loss for Face Recognition
1. Arxiv 2019
2. Wang, Xiaobo and Wang, Shuo and Zhang, Shifeng and Fu, Tianyu and Shi, Hailin and Mei, Tao
3. 190211(4)Support Vector Guided Softmax Loss for Face Recognition.pdf

- Face recognition has witnessed significant progresses
due to the advances of deep convolutional neural networks
(CNNs), the central challenge of which, is feature discrim-
ination. To address it, one group tries to exploit mining-
based strategies (e.g., hard example mining and focal loss)
to focus on the informative examples. The other group de-
votes to designing margin-based loss functions (e.g., angu-
lar, additive and additive angular margins) to increase the
feature margin from the perspective of ground truth class.
Both of them have been well-verified to learn discrimina-
tive features.

    ![reid](Pictures/Selection_353.png)

- However, they suffer from either the ambigu-
ity of hard examples or the lack of discriminative power of
other classes. In this paper, we design a novel loss function,
namely support vector guided softmax loss (SV-Softmax),
which adaptively emphasizes the mis-classified points (sup-
port vectors) to guide the discriminative features learning.
So the developed SV-Softmax loss is able to eliminate the
ambiguity of hard examples as well as absorb the discrimi-
native power of other classes, and thus results in more dis-
crimiantive features. To the best of our knowledge, this is
the first attempt to inherit the advantages of mining-based
and margin-based losses into one framework. Experimental
results on several benchmarks have demonstrated the effec-
tiveness of our approach over state-of-the-arts.


>@article{wang2018support,
  title={Support Vector Guided Softmax Loss for Face Recognition},
  author={Wang, Xiaobo and Wang, Shuo and Zhang, Shifeng and Fu, Tianyu and Shi, Hailin and Mei, Tao},
  journal={arXiv preprint arXiv:1812.11317},
  year={2018}
}
