# Moving in the Right Direction: A Regularization for Deep Metric Learning


This repository is implementation of CVPR paper written by Deen Dayal Mohan et al. 
Refer to [the paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohan_Moving_in_the_Right_Direction_A_Regularization_for_Deep_Metric_CVPR_2020_paper.pdf) for detailed explanation. 


# Table of Contents
* [Advantages of Regularization](#advantages-of-regularization)
* [Qualitative Results](#qualitative-results)
* [Quantitative Results](#quantitative-results)
* [ToDo List](#todo-list)
* [Training instructions](#training-instructions)


# Advantages of Regularization
1. In vanilla triplet loss, negative sample is pushed away from only anchor example so vanilla triplet do not exploit from positive examples in batch. The regularization constraints movement of the negative sample through direction that is perpendicular to line segment between anchor and positive.
2. Regularization provides inherent mining mechanism to prevent negative samples ,that are near both positive and anchor, for being used in parameter update.

## Qualitative Results

## Quantitative Results
| Recall | 1 | 2 | 4 | 8 |
|--------|---|---|---|---|
| Triplet|   |   |   |   | 
| DR-Triplet|  |  |  |   |
| Proxy | | | | |
| DR-Proxy | | | | |

## Tuned Parameter
Optuna framework is utilized to tune hyper-parameter of the method.

## ToDo List
### Alper
- [x] Triplet Loss with Regularizer
- [x] Proxy Loss with Regularizer
- [x] Implementation of Recall Metric
- [x] Visualization for Qualitative Results
- [x] CUB-Dataset 
- [x] Data Transformations
- [x] Hyper parameter tuning for triplet and proxy
- [x] Main codes for train and test

### Baran


## Training instructions
