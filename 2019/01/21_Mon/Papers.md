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