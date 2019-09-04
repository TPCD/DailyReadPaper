# DailyReadPaper
Catch up with the state of the art! These are latest paper about Re-ID, which are collected from Arxiv and ICCV 2019.

## Weakly Supervised Person Re-Identification
- In the conventional person re-id setting, it is assumed
that the labeled images are the person images within the
bounding box for each individual; this labeling across multiple nonoverlapping camera views from raw video surveillance is costly and time-consuming. To overcome this difficulty, we consider weakly supervised person re-id modeling.
The weak setting refers to matching a target person with an
untrimmed gallery video where we only know that the identity appears in the video without the requirement of annotating the identity in any frame of the video during the training procedure. Hence, for a video, there could be multiple
video-level labels. We cast this weakly supervised person
re-id challenge into a multi-instance multi-label learning
(MIML) problem. In particular, we develop a Cross-View
MIML (CV-MIML) method that is able to explore potential
intraclass person images from all the camera views by incorporating the intra-bag alignment and the cross-view bag
alignment. Finally, the CV-MIML method is embedded into
an existing deep neural network for developing the Deep
Cross-View MIML (Deep CV-MIML) model. We have performed extensive experiments to show the feasibility of the
proposed weakly supervised setting and verify the effectiveness of our method compared to related methods on four
weakly labeled datasets.

    ![Arxiv](Pictures/1.png)
    
    

>inproceedings{fmeng2019weakly,
title=fWeakly Supervised Person Re-Identificationg,
author=fMeng, Jingke and Wu, Sheng and Zheng, Wei-Shig,
booktitle=fProceedings of the IEEE International Conference on Computer Vision and Pattern Recognitiong,
year=f2019
}
## Visual Person Understanding through Multi-Task and Multi-Dataset Learning
1. Arxiv 2019
2. Kilian Pfeiffer, Alexander Hermans, Istv´an S´ar´andi, Mark Weber, and Bastian Leibe

- We address the problem of learning a single model for person
re-identification, attribute classification, body part segmentation, and
pose estimation. With predictions for these tasks we gain a more holistic
understanding of persons, which is valuable for many applications. This
is a classical multi-task learning problem. However, no dataset exists that
these tasks could be jointly learned from. Hence several datasets need
to be combined during training, which in other contexts has often led to
reduced performance in the past. We extensively evaluate how the different task and datasets influence each other and how different degrees
of parameter sharing between the tasks affect performance. Our final
model matches or outperforms its single-task counterparts without creating significant computational overhead, rendering it highly interesting
for resource-constrained scenarios such as mobile robotics.. 

    ![Arxiv](Pictures/2.png)
    
## Universal Person Re-Identification
1. Arxiv 2019
2. Xu Lan Queen Mary University of London，Xiatian Zhu Vision Semantics Ltd.，Shaogang Gong Queen Mary University of London
- Most state-of-the-art person re-identification (re-id) methods
depend on supervised model learning with a large set of crossview identity labelled training data. Even worse, such trained
models are limited to only the same-domain deployment with
significantly degraded cross-domain generalisation capability, i.e. “domain specific”. To solve this limitation, there are
a number of recent unsupervised domain adaptation and unsupervised learning methods that leverage unlabelled target
domain training data.

    ![Arxiv](Pictures/3.png)
    
- However, these methods need to train
a separate model for each target domain as supervised learning methods. This conventional “train once, run once” pattern is unscalable to a large number of target domains typically encountered in real-world deployments. We address this
problem by presenting a “train once, run everywhere” pattern
industry-scale systems are desperate for. We formulate a “universal model learning” approach enabling domain-generic
person re-id using only limited training data of a “single” seed
domain. Specifically, we train a universal re-id deep model
to discriminate between a set of transformed person identity
classes. Each of such classes is formed by applying a variety
of random appearance transformations to the images of that
class, where the transformations simulate the camera viewing conditions of any domains for making the model training domain generic. Extensive evaluations show the superiority of our method for universal person re-id over a wide variety of state-of-the-art unsupervised domain adaptation and
unsupervised learning re-id methods on five standard benchmarks: Market-1501, DukeMTMC, CUHK03, MSMT17, and
VIPeR.

## Temporal Knowledge Propagation for Image-to-Video Person Re-identification
1. Arxiv 2019
2. Xinqian Gu1;2, Bingpeng Ma2, Hong Chang1;2, Shiguang Shan1;2;3, Xilin Chen1;2

- In many scenarios of Person Re-identification (Re-ID),
the gallery set consists of lots of surveillance videos and the
query is just an image, thus Re-ID has to be conducted between image and videos. Compared with videos, still person
images lack temporal information. Besides, the information
asymmetry between image and video features increases the
difficulty in matching images and videos. To solve this problem, we propose a novel Temporal Knowledge Propagation
(TKP) method which propagates the temporal knowledge
learned by the video representation network to the image
representation network.

    ![Arxiv](Pictures/4.png)
    
- Specifically, given the input videos,
we enforce the image representation network to fit the outputs of video representation network in a shared feature
space. With back propagation, temporal knowledge can
be transferred to enhance the image features and the information asymmetry problem can be alleviated. With additional classification and integrated triplet losses, our model
can learn expressive and discriminative image and video
features for image-to-video re-identification.

    ![Arxiv](Pictures/6.png)
    
- Extensive experiments demonstrate the effectiveness of our method and
the overall results on two widely used datasets surpass the
state-of-the-art methods by a large margin.

## Non-local Neural Networks
1. CVPR 2018
2. Xiaolong Wang1,2∗ Ross Girshick2 Abhinav Gupta1 Kaiming He2

- Both convolutional and recurrent operations are building
blocks that process one local neighborhood at a time. In
this paper, we present non-local operations as a generic
family of building blocks for capturing long-range dependencies. Inspired by the classical non-local means method
[4] in computer vision, our non-local operation computes
the response at a position as a weighted sum of the features
at all positions.

    ![Arxiv](Pictures/5.png)
    
- This building block can be plugged into
many computer vision architectures. On the task of video
classification, even without any bells and whistles, our nonlocal models can compete or outperform current competition
winners on both Kinetics and Charades datasets. In static
image recognition, our non-local models improve object detection/segmentation and pose estimation on the COCO suite
of tasks. Code will be made available.


## Spatially and Temporally Efficient Non-local Attention Network for Video-based Person Re-Identification
1. BMVC 2019
2. National Taiwan University, Taiwan
- Video-based person re-identification (Re-ID) aims at matching video sequences of
pedestrians across non-overlapping cameras.

    ![Arxiv](Pictures/7.png)
    
- It is a practical yet challenging task of
how to embed spatial and temporal information of a video into its feature representation. While most existing methods learn the video characteristics by aggregating imagewise features and designing attention mechanisms in Neural Networks, they only explore the correlation between frames at high-level features. In this work, we target
at refining the intermediate features as well as high-level features with non-local attention operations and make two contributions. (i) We propose a Non-local Video Attention Network (NVAN) to incorporate video characteristics into the representation at
multiple feature levels. (ii) We further introduce a Spatially and Temporally Efficient
Non-local Video Attention Network (STE-NVAN) to reduce the computation complexity by exploring spatial and temporal redundancy presented in pedestrian videos. Extensive experiments show that our NVAN outperforms state-of-the-arts by 3:8% in rank-
1 accuracy on MARS dataset and confirms our STE-NVAN displays a much superior
computation footprint compared to existing methods. Codes are available at https:
//github.com/jackie840129/STE-NVAN.


