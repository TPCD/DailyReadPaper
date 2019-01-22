# DailyReadPaper
- Zero-shot learning is a promising aspect for solving the open problems (real scenario).
Today I will read some paper about ZSL and deliver them to this website.
Meanwhile, for the purposes of strengthening memory and deepening understanding paper I've read, some 
of these papers were elaborated as follows:

## Dissimilarity Representation Learning for Generalized Zero-Shot Recognition
1. ACM MM 2018
2. Yang, Gang and Liu, Jinlu and Xu, Jieping and Li, Xirong

- Generalized zero-shot learning (GZSL) aims to recognize any test
instance coming either from a known class or from a novel class
that has no training instance.

    ![GZSL](Pictures/1.png)
    
- To synthesize training instances for
novel classes and thus resolving GZSL as a common classifcation problem, we propose a Dissimilarity Representation Learning
(DSS) method. 

    ![GZSL](Pictures/2.png)
    
- Dissimilarity representation is to represent a specifc instance in terms of its (dis)similarity to other instances in a
visual or attribute based feature space. 

    ![GZSL](Pictures/3.png)
    ![GZSL](Pictures/4.png) 
    
- In the dissimilarity space, instances of the novel classes are synthesized by an end-to-end
optimized neural network. 

    ![GZSL](Pictures/5.png) 
    ![GZSL](Pictures/6.png)
        
- The neural network realizes two-level feature mappings and domain adaptions in the dissimilarity space
and the attribute based feature space. Experimental results on fve
benchmark datasets, i.e., AWA, AWA2, SUN, CUB, and aPY, show
that the proposed method improves the state-of-the-art with a large
margin, approximately 10% gain in terms of the harmonic mean
of the top-1 accuracy. Consequently, this paper establishes a new
baseline for GZSL.

>@inproceedings{yang2018dissimilarity,
  title={Dissimilarity Representation Learning for Generalized Zero-Shot Recognition},
  author={Yang, Gang and Liu, Jinlu and Xu, Jieping and Li, Xirong},
  booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
  pages={2032--2039},
  year={2018},
  organization={ACM}
}

# A rethink about A Riemannian Network for SPD Matrix Learning
- The significant contributions in this paper include several point as follows,
    1. They propose a new way of updating the weights for the BiMap layers by
    exploiting an SGD setting **on Stiefel manifolds.** As
    merely computing their Euclidean gradients in the procedure of 
    backprop cannot valid orthogonal weights, they have to force the 
    weights to be on Stiefel manifolds. Subsequently, in order to obtain the Riemannian gradient, **the normal component of the Euclidean gradient is subtracted
    to generate the tangential component to the Stiefel manifold.** 
    2. Because computing those with EIG decomposition in the layers of ReEig and LogEig has not been
    well-solved by the traditional backprop. They exploit the matrix generalization of backprop studied in (Ionescu, Vantzos, and Sminchisescu 2015) to compute the gradients of the involved
    SPD matrices in the ReEig and LogEig layers. In particular, **let F be a function describing the variations of the upper
    layer variables with respect to the lower layer variables**. With the function F, a new version of
    the chain rule Eqn.6 for the matrix backprop.

# The reference in A Riemannian Network for SPD Matrix Learning
## Covariance Discriminative Learning: A Natural and Efficient Approach to Image Set Classification
- We propose a novel discriminative learning approach to
image set classification by **modeling the image set with its
natural second-order statistic**, i.e. covariance matrix. Since
nonsingular covariance matrices, a.k.a. symmetric positive
definite (SPD) matrices, **lie on a Riemannian manifold**,
classical learning algorithms cannot be directly utilized to
classify points on the manifold. By exploring an **efficient
metric for the SPD matrices**, i.e., Log-Euclidean Distance
(LED), we **derive a kernel function** that explicitly **maps** the
**covariance matrix** from the **Riemannian manifold** to a
**Euclidean space**. 

    ![GZSL](Pictures/7.png)
    
- With this explicit mapping, any learning
method devoted to vector space can be exploited in either
its linear or kernel formulation. Linear Discriminant
Analysis (LDA) and Partial Least Squares (PLS) are
considered in this paper for their feasibility for our specific
problem.
- We further investigate the conventional linear
subspace based set modeling technique and cast it in a
unified framework with our covariance matrix based
modeling. The proposed method is evaluated on two tasks:
face recognition and object categorization. 

>@inproceedings{wang2012covariance,
  title={Covariance discriminative learning: A natural and efficient approach to image set classification},
  author={Wang, Ruiping and Guo, Huimin and Davis, Larry S and Dai, Qionghai},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on},
  pages={2496--2503},
  year={2012},
  organization={IEEE}
} 