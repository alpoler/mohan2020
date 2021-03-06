# Moving in the Right Direction: A Regularization for Deep Metric Learning


This repository is implementation of CVPR paper written by Deen Dayal Mohan et al. 
Refer to [the paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohan_Moving_in_the_Right_Direction_A_Regularization_for_Deep_Metric_CVPR_2020_paper.pdf) for detailed explanation. 


# Table of Contents
* [Advantages of Regularization](#advantages-of-regularization)
* [Qualitative Results](#qualitative-results)
* [Quantitative Results](#quantitative-results)
* [ToDo List](#todo-list)
* [instructions](#instructions)
* [Contact](#contact)


# Advantages of Regularization
1. In vanilla triplet loss, negative sample is pushed away from only anchor example so vanilla triplet do not exploit from positive examples in batch. The regularization constraints movement of the negative sample through direction that is perpendicular to line segment between anchor and positive.
2. Regularization provides inherent mining mechanism to prevent negative samples ,that are near both positive and anchor, for being used in parameter update.

## Qualitative Results

![kuslar3](https://user-images.githubusercontent.com/50836811/126173980-a3d4258d-bf27-4077-b0f3-8206831b0285.png)


## Quantitative Results
| Recall | 1 | 2 | 4 | 8 |
|--------|---|---|---|---|
| Triplet|  51.9 | 64.0 | 70.3  | 74.1 | 
| DR-Triplet| 54.49 | 66.22 | 77.5 | 85.79 |
| ProxyNCA | 49.2 |61.9 | 67.90 | 72.4 |
| DR-ProxyNCA | 52.43 | 63.74 | 74.05 | 83.37 |


## Tuned Parameter


Optuna framework is utilized to tune hyper-parameter of the method. 
Triplet Hyper-Parameter 
```
Margin : 0.2781877469005122 
Reg. Constant: 0.4919607680052035
Learning Rate: 1e-5
Patience: 25
Batch size: 128 
```
Proxy Hyper-Parameter
```
Batch size: 196
Learning Rate: 1e-4
Patience: 20
```
Model is trained with hyper-parameter corresponding to the specified loss together with fixed hyper-parameter mentioned below for reproducibility.
```
Emb. Dim: 64 
Optimizer: Adam 
```

## Instructions
1. Download dataset and put into folder named 'data'.
2.
DR-TRIPLET:
```
python train.py --batch_size 128 --patience 25 --mvr_reg 0.4919607680052035 --margin 0.2781877469005122 --loss mvr_triplet --tnsrbrd_dir ./runs/exp_trp --model_save_dir ./MVR_Triplet/exp  --exp_name mvr_triplet
```
DR-PROXYNCA:
```
python train.py --batch_size 128 --patience 25 --mvr_reg 0.45 --loss mvr_proxy --tnsrbrd_dir ./runs/exp_proxy --model_save_dir ./MVR_Proxy/exp --exp_name mvr_proxy 
```
For visualization
Create folder with name you desired inside log directory. Please change name of 'proxy_exp20' with name you assing for log folder. 

DR-Triplet:
```
python test.py --exp_name mvr_triplet --model_save_dir ./MVR_Triplet/exp
```
DR-PROXYNCA:
```
python test.py --exp_name mvr_proxy --model_save_dir ./MVR_Proxy/exp
```

## Contact
Feel free to ask me any question. Email address: alper.kayabasi@metu.edu.tr, [Linkedin](https://www.linkedin.com/in/kayabasialper/)
