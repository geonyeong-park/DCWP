# Training Debiased Subnetworks with Contrastive Weight Pruning

[![arXiv](https://img.shields.io/badge/arXiv-2210.05247-b31b1b.svg)](https://arxiv.org/abs/2210.05247)

## Abstract
> Neural networks are often biased to spuriously correlated features that provide misleading statistical evidence that does not generalize. 
This raises an interesting question: "Does an optimal unbiased functional subnetwork exist in a severely biased network? If so, how to extract such subnetwork?" 
While empirical evidence has been accumulated about the existence of such unbiased subnetworks, these observations are mainly based on the guidance of ground-truth unbiased samples. 
Thus, it is unexplored how to discover the optimal subnetworks with biased training datasets in practice. To address this, here we first present our theoretical insight that alerts potential limitations of existing algorithms in exploring unbiased subnetworks in the presence of strong spurious correlations. 
We then further elucidate the importance of bias-conflicting samples on structure learning. 
Motivated by these observations, we propose a Debiased Contrastive Weight Pruning (DCWP) algorithm, which probes unbiased subnetworks without expensive group annotations. 
Experimental results demonstrate that our approach significantly outperforms state-of-the-art debiasing methods despite its considerable reduction in the number of parameters.

![concept](dataset/concept.png)

## Prerequisites
- python 3.8
- pytorch >= 1.13.1
- CUDA 11.6

It is okay to use a lower version of CUDA with a proper pytorch version.

## Getting started

### 1. Clone the repository
```
git clone https://github.com/ParkGeonYeong/DCWP.git
cd DCWP
```

### 2. Set the environment
```
conda env create --name DCWP --file env.yaml
conda activate DCWP
```

### 3. Dataset

![sample](dataset/sample_figure.png)

We use four different biased datasets: CMNIST, CIFAR10-C, BFFHQ and CelebA (blonde). Every data files should be saved in the `dataset/{dataset_name}` folder.
- CMNIST, CIFAR10-C and BFFHQ: download the datasets [here](https://drive.google.com/drive/folders/1JEOqxrhU_IhkdcRohdbuEtFETUxfNmNT)
which comes from [DisEnt](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled).

- CelebA: download from [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset). In `dataset/celebA`, `metadata_blonde_subsampled.csv` denotes the metadata of each image, e.g. label, split, etc. 
- Sample images are presented in each `dataset/{dataset_name}` folder. After download, the images should be saved in `dataset` following the below structure:

```
dataset/cmnist, cifar10c 
 └ 0.5pct / 1pct / 2pct / 5pct
     └ align
     └ conlict
     └ valid
 └ test
```

```
dataset/bffhq
 └ 0.5pct
 └ valid
 └ test
```

```
dataset/celebA
 └ celeba
    └ img_align_celeba
 └ metadata_blonde_subsampled.csv
```

## Main scripts
We provide multiple bash scripts for each dataset. 
```
bash train_ours_{dataset}.sh {gpu} 
```
where `dataset` corresponds to one of _cmnist, cifar10c, bffhq, celeba_. `gpu` indicates the index of GPU, e.g. 0.
In general, (1) pre-training, (2) mining, (3) pruning, and (4) fine-tuning can be executed in a single process without termination.
For CMNIST, (1) and (2) are conducted with SGD, and (3) and (4) are conducted with Adam. 

Some of the key arguments are summarized below:
- `--mode`: `prune` corresponds to the proposed framework.
- `--conflict_pct`: Percent of bias-conflicting samples (for CMNIST, CIFAR10-C).
- `--lambda_upweight`: $\lambda_{up}$ in $\ell_{WCE}$.
- `--lambda_sparse`: $\lambda_{\ell_1}$ in main objective, eq (11).
- `--pseudo_label_method`: Bias-conflicting mining algorithms.
   - `wrong` treats the samples misclassified by the biased model as bias-conflicting proxies. If `--select_with_GCE`, the biased model is trained with GCE.
   - `ensemble` refers to the algorithms modified from [here](https://arxiv.org/abs/2111.13108). Only used for CIFAR10-C.
- `--lr_pre, lr_main`: Learning rate for pre-training and fine-tuning.
- `--pretrain_iter, retrain_iter`: Number of pre-training and fine-tuning iterations.
- `--lr_decay_step_pre, lr_decay_step_main`: Learning rate decay step. Usually we did not use learning rate decaying, i.e. `lr_decay_step_pre == pretrain_iter`.
- `--imagenet`: Initialize from ImageNet-pretrained weights.
