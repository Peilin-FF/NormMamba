# Mamba: Long Range Arena

This repository contains experiments for Mamba and its performance on the [long range arena (LRA)](https://arxiv.org/abs/2011.04006) tasks, as well as a few other experiments related to Mamba. 

A lot of the LRA code is taken from the repository [not-from-scratch](https://github.com/IdoAmos/not-from-scratch/) and [Long-Range-Arena](https://github.com/google-research/long-range-arena). We furthermore use the [Mamba](https://github.com/state-spaces/mamba/tree/main) block from the [Mamba paper](https://arxiv.org/abs/2312.00752).

# More Information:

A comprehensive summary of our results can be found in the [article](https://github.com/fluderm/MLRA/blob/main/Mamba_report.pdf) and a very brief summary in the [presentation slides](https://github.com/fluderm/MLRA/blob/main/Mamba_slides.pdf).  

# Prepare Data in data folder
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
gunzip /data/vlm/fpl/MLRA/data/lra_release.gz
tar -xvf /data/vlm/fpl/MLRA/data/lra_release -C /data/vlm/fpl/MLRA/data/

# Environment
Attention: We use python==3.11,torch==2.0.0,cuda==11.7,mamba-ssm==2.2.0


conda create -n MLRA python==3.11
conda activate MLRA

torch==2.0.0
torchaudio==2.0.1
torchdata==0.6.0
torchmetrics==1.4.2
torchtext==0.15.1
torchvision==0.15.1
pytorch-lightning==2.2.0
rotary-embedding-torch==0.8.3
conda install -c conda-forge cudatoolkit=11.7

start:
CUDA_VISIBLE_DEVICES={7} python train.py experiment=lra/mamba-lra-listops

project: state-spaces