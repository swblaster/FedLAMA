
# FedLAMA
This repository contains the S/W framework used for all the experiments in the below paper.
'[Sunwoo Lee, Tuo Zhang, and Salman Avestimehr, Layer-wise Adaptive Model Aggregation for Scalable Federated Learning, AAAI, 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26023)'

## Software Requirements
 * tensorflow2 (<= 2.15.1)
 * tensorflow_datasets
 * python3
 * mpi4py
 * tqdm

## Instructions
### Training
 1. Set hyper-parameters properly in `config.py`.
 2. Put the dataset files in the top directory of this program. The directory name should be the same as `dataset` in config.py.
 3. Run training.
```
mpiexec -n 8 python main.py
```
### Output
This program evaluates the trained model after every epoch and then outputs the results as follows.
 1. `loss.txt`: An output file that contains the training loss for every epoch.
 2. `acc.txt`: An output file that contains the validation accuracy for every epoch.
 3. `./checkpoint`: The checkpoint files generated after every epoch. This directory is created only when `checkpoint` is set to 1 in `config.py`.

## Results
We will provide a few key experimental results here once the papers are published.

## Supported Federated Learning Features
 * FedAvg
 * FedLAMA

## Supported Datasets
 * CIFAR-10
 * CIFAR-100

## Questions / Comments
 * Sunwoo Lee (sunwool@inha.ac.kr)
