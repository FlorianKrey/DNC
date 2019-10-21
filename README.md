# Discriminative Neural Clustering (DNC) for Speaker Diarisation

This repository is the code used in our paper:

>**[Discriminative Neural Clustering for Speaker Diarisation](https://arxiv.org/abs/abcd.efghi)**
>
>*Qiujia Li\*, Florian Kreyssig\*, Chao Zhang, Phil Woodland* (\* indicates equal contribution)
>
>Submitted to ICASSP 2020

## Overview
We propose to use encoder-decoder models for supervised clustering. This repository contains:
* a submodule for spectral clustering, a modified version of [this repository by Google](https://github.com/wq2012/SpectralCluster)
* a submodule for DNC using Transformers, implemented in [ESPnet](https://github.com/espnet/espnet)
* data processing procedures for data augmentation & curriculum learning in our paper

## Dependencies
First, as this repository contains two submodules, please clone this repository using
```bash
git clone --recursive https://github.com/FlorianKrey/DNC.git
```
Then execute the following command to install MiniConda for virtualenv with related packages:
```bash
cd DNC
./install.sh
```
Note that you may want to change the CUDA version for PyTorch according to your own driver.

## Data generation

## Training and decoding of DNC models

## Running spectral clustering

## Evaluation of clustering results

## Reference
```plaintext
@article{LiKreyssig2019DNC,
  title={Discriminative Neural Clustering for Speaker Diarisation},
  author={Li, Qiujia and Kreyssig, Florian L. and Zhang, Chao and Woodland, Philip C.},
  journal={ArXiv.org},
  year={2019},
  volume={abs/abcd.efghi},
  url={http://arxiv.org/abs/abcd/efghi}
}
```
