# Paper title

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

This repository is the implementation of the CVPR 2020 paper written by Deen Dayal Mohan et al. Please refer to [the paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Mohan_Moving_in_the_Right_Direction_A_Regularization_for_Deep_Metric_CVPR_2020_paper.html) for a detailed explanation. Moving in Right Direction (MVR) paper tries to solve the metric learning problem, which aims to construct embedding space where distance is inversely correlated with semantic similarity. Metric learning has many practical applications such as image retrieval, face verification, person re-identification, few-shot learning, etc. Image retrieval aims to bring samples as same class as the given query image. MVR paper introduces direction regularization to prevent the unaware movement of samples. Therefore, their methodology improves the performance of retrieval tasks compared to a method without regularization. We aim to quantitatively validate retrieval performance increases and visualize retrieval results to see the capability of deep metric learning during this repository.

## 1.1. Paper summary

Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method

Explain the original method.

## 2.2. My interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

As model, MVR paper utilizes pretrained GoogleNet with Batch Normalization on ImageNet. Although they do not express which pretrained model they use, we choose caffe pretrained model due to superiority over pytorch pretrained model. The caffe model only perform zero mean preprocessing to the dataset compared to torch model that applies not only zero mean but also scaling of the dataset as a preprocessing. As mentioned in the paper, we augment train dataset with random cropping and random horizontal flip while test set is center cropped. We evaluate performance on CUB-200-2011 dataset but it is easily generalizable to other dataset. CUB dataset is split into two equal part as train and test set in the MVR paper ;however, they do not mention existence of validation set. Therefore, we assume that they do not use validation set. This fact is mentioned in Metric Learning Reality Check paper that majority of paper do not use validation set.

## 3.2. Running the code

Explain your code & directory structure and how other people can run it.

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

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

Provide your references here.

# Contact

Alper Kayabasi - alperkayabasi97@gmail.com
Baran Gulmez - baran.gulmez07@gmail.com
