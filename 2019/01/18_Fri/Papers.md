# DailyReadPaper

## Salient Object Detection via High-to-Low Hierarchical Context Aggregation
1. arXiv:1812.10956 (Submitted on 28 Dec 2018)
2. Yun Liu, Yu Qiu, Le Zhang, JiaWang Bian, Guang-Yu Nie, **Ming-Ming Cheng**
3. https://github.com/yun-liu/HCA

- From the perspective of author, **fusion strategies** should not be more and 
more **complex** for accurate salient object detection.
Thus, they design an Mirror-linked Hourglass Network with intermediate supervision to learn
 contextual features in a **high-to-low** manner. 
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_003.png)

- The contexts of a natural image can be well expressed by a **high-to-low self-learning of side-output
convolutional features** as follows, 
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_005.png)

- Through a Hierarchical Context Aggregation (HCA) module, the learned hierarchical contexts are aggregated to generate the hybrid 
contextual expression for an input image. 
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_004.png)

- Their **simple** method achieves state-of-the-art 
performance under various evaluation metrics.

>@misc{1812.10956,
Author = {Yun Liu and Yu Qiu and Le Zhang and JiaWang Bian and Guang-Yu Nie and Ming-Ming Cheng},
Title = {Salient Object Detection via High-to-Low Hierarchical Context Aggregation},
Year = {2018},
Eprint = {arXiv:1812.10956}}

## Deeply Supervised Salient Object Detection with Short Connections
1. CVPR 2017
2. Hou, Qibin and **Cheng, Ming-Ming** and Hu, Xiaowei and Borji, Ali and Tu, Zhuowen and Torr, Philip HS
3. https://mmcheng.net/zh/code-data/

- From the perspective of author, there is still a large room for improvement over the **generic** FCN models 
that do not explicitly deal with the scale-space problem. Holistically-Nested Edge Detector (HED) provides a 
skip-layer structure with deep supervision for edge and boundary detection, but the performance gain of HED
on saliency detection is not obvious. 
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_007.png)

- They propose a new salient object detection method 
by introducing short connections to the skip-layer structures within the HED
architecture. 
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_006.png)

>@inproceedings{hou2017deeply,
  title={Deeply supervised salient object detection with short connections},
  author={Hou, Qibin and Cheng, Ming-Ming and Hu, Xiaowei and Borji, Ali and Tu, Zhuowen and Torr, Philip HS},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3203--3212},
  year={2017}}


## Detect globally, refine locally: A novel approach to saliency detection
1. CVPR 2018
2. Wang, Tiantian and Zhang, Lihe and Wang, Shuo and Lu, Huchuan and Yang, Gang and Ruan, Xiang and Borji, Ali
3. https://github.com/TiantianWang/CVPR18_detect_globally_refine_locally

- Effective integration of contextual information is crucial for salient object detection. To achieve this, most existing methods based on ’skip’ architecture mainly focus on how to integrate hierarchical features of Convolutional Neural Networks (CNNs). They simply apply concatenation or element-wise operation to incorporate high-level
semantic cues and low-level detailed information. However, this can degrade the quality of predictions because cluttered and noisy information can also be passed through.
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_010.png)

- To address this problem, we proposes a global Recurrent Localization Network (RLN) which exploits contextual information by the weighted response map in order to localize salient objects more accurately. 
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_011.png)

- Particularly, a recurrent module is employed to progressively refine the inner structure of the CNN over multiple time steps. 
   ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_012.png)
 
- Moreover, to effectively recover object boundaries, we propose a local Boundary Refinement Network (BRN) to adaptively learn
the local contextual information for each spatial position.
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_013.png)

- The learned propagation coefficients can be used to optimally capture relations between each pixel and its neighbors. Experiments on five challenging datasets show that our approach performs favorably against all existing methods in terms of the popular evaluation metrics.


>@inproceedings{wang2018detect,
  title={Detect globally, refine locally: A novel approach to saliency detection},
  author={Wang, Tiantian and Zhang, Lihe and Wang, Shuo and Lu, Huchuan and Yang, Gang and Ruan, Xiang and Borji, Ali},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3127--3135},
  year={2018}}

## Amulet: Aggregating multi-level convolutional features for salient object detection
1. ICCV 2017
2. Wang, Tiantian and Zhang, Lihe and Wang, Shuo and Lu, Huchuan and Yang, Gang and Ruan, Xiang and Borji, Ali
3. https://github.com/Pchank/caffe-sal

- Fully convolutional neural networks (FCNs) have shown
outstanding performance in many dense labeling problems.
One key pillar of these successes is mining relevant information from features in convolutional layers. However, how
to better aggregate multi-level convolutional feature maps
for salient object detection is underexplored. In this work,
we present Amulet, a generic aggregating multi-level con-
volutional feature framework for salient object detection.
Our framework first integrates multi-level feature maps in-
to multiple resolutions, which simultaneously incorporate
coarse semantics and fine details. Then it adaptively learns
to combine these feature maps at each resolution and predict saliency maps with the combined features. Finally, the
predicted results are efficiently fused to generate the final
saliency map. 
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_009.png)

- In addition, to achieve accurate boundary
inference and semantic enhancement, edge-aware feature
maps in low-level layers and the predicted results of low
resolution features are recursively embedded into the learn-
ing framework. 
    ![Ming-Ming Cheng2018 Salient Object Detection](Pictures/Selection_008.png)

- By aggregating multi-level convolutional
features in this efficient and flexible manner, the proposed
saliency model provides accurate salient object labeling.
Comprehensive experiments demonstrate that our method
performs favorably against state-of-the-art approaches in
terms of near all compared evaluation metrics.
>@inproceedings{zhang2017amulet,
  title={Amulet: Aggregating multi-level convolutional features for salient object detection},
  author={Zhang, Pingping and Wang, Dong and Lu, Huchuan and Wang, Hongyu and Ruan, Xiang},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={202--211},
  year={2017}}