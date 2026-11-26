# Feature-Space-Regularized Personalized Federated Learning via Prototypical Distribution

This repository contains the code and data used to obtain Tables 1, 2, and 3 in the manuscript. The main entry point is `exp_main.py`. One execution corresponds to one random seed and costs about 10 hours on a single NVIDIA RTX 4090 GPU. The results will be saved in the folder **[`eval/`](./eval/)**.

## Code Organization

- **[`data/`](./data/)**: Contains dataset processing and loading utilities, along with the MNIST datasets.
- **[`utils/`](./utils/)**: Provides utility functions for model inference, matrix operations, performance evaluation, hyperparameter configuration, and other common operations.
- **[`network/`](./network/)**: Contains neural network architectures used for feature extraction and classification.
- **[`src/`](./src/)**: Implements the core federated learning framework, including client-side training and server-side aggregation.

## Environment

Install the required dependencies with:

```bash
pip install -r requirement.txt
```

## How to run the experiment

Run the balanced experiments with seed 1:

```bash
python exp_main.py --dataset RotatedMNIST --seed 1
python exp_main.py --dataset ColoredMNIST --seed 1
```

Run the class-imbalanced experiments with seed 1:

```bash
python exp_main.py --dataset RotatedMNIST --seed 1 --imbalance-ratio 0.005
python exp_main.py --dataset ColoredMNIST --seed 1 --imbalance-ratio 0.005
```
