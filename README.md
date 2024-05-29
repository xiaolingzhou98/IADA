
# Boosting Model Resilience via Implicit Adversarial Data Augmentation

 
This repository contains the code of "Boosting Model Resilience via Implicit Adversarial Data Augmentation (IJCAI 2024)".

#### Citation
If you find this study useful for your research, please cite our paper.
```
@article{zhou2024boosting,
  title={Boosting Model Resilience via Implicit Adversarial Data Augmentation},
  author={Zhou, Xiaoling and Ye, Wei and Lee, Zhemg and Xie, Rui and Zhang, Shikun},
  journal={arXiv preprint arXiv:2404.16307},
  year={2024}
}
```

 
#### Abstract

Data augmentation plays a pivotal role in enhancing and diversifying training data. Nonetheless, consistently improving model performance in varied learning scenarios, especially those with inherent data biases, remains challenging. To address this, we propose to augment the deep features of samples by incorporating their adversarial and anti-adversarial perturbation distributions, enabling adaptive adjustment in the learning difficulty tailored to each sample’s specific characteristics. We then theoretically reveal that our augmentation process approximates the optimization of a surrogate loss function as the number of augmented copies increases indefinitely. This insight leads us to develop a meta-learning-based framework for optimizing classifiers with this novel loss, introducing the effects of augmentation while bypassing the explicit augmentation process. We conduct extensive experiments across four common biased learning scenarios: long-tail learning, generalized long-tail learning, noisy label learning, and subpopulation shift learning. The empirical results demonstrate that our method consistently achieves state-of-the-art performance, highlighting its broad adaptability.

## Requirements

- PyTorch >= 1.2.0
- Python3
- torchvision
- PIL
- argparse
- numpy


## Getting Started

### Dataset
- Long-tailed CIFAR10/100: The long-tailed version of CIFAR10/100. 
- Noisy CIFAR10/100: The noisy version of CIFAR10/100. 
- ImageNet-LT: The long-tailed version of ImageNet.
- Places-LT: The long-tailed version of Places-2.
- [iNaturalist2018](https://github.com/visipedia/inat_comp/tree/master/2018): A natural long-tailed dataset.

### Train

Training on CIFAR-LT-10/100:
```
CIFAR100-LT, 
python MetaIADA.py --imb_factor 10 --dataset cifar100 --num_classes 100
```

Or run the script:

```
sh train.sh
```

Training on Noisy CIFAR10/100 with flip noise:
```
Noisy CIFAR100, 
python MetaIADA.py --corruption_type flip2 --corruption_ratio 0.2 --dataset cifar100 --num_classes 100
```


Or run the script:

```
sh train-noise.sh
```

## Acknowledgements
Some codes in our project are adapted from [MetaSAug](https://github.com/BIT-DA/MetaSAug) and [Meta-Weight-Net](https://github.com/ShiYunyi/Meta-Weight-Net_Code-Optimization). We express our gratitude for their outstanding projects.
