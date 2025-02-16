import os

import pynever.datasets as dt
import torch
from torchvision import transforms as tr

from linearverifier.core import ops


def test_accuracy(weights: str, bias: str) -> float:
    with open(weights, 'r') as file:
        lines = file.readlines()
        weights_list = [[float(v) for v in line.strip('\n').split(',')] for line in lines]

    with open(bias, 'r') as file:
        lines = file.readlines()
        bias_list = [float(line.strip('\n')) for line in lines]

    # Load dataset
    test_set = dt.TorchMNIST('./../../../Data', train=False,
                             transform=tr.Compose([tr.ToTensor(), tr.Lambda(lambda x: torch.flatten(x))]),
                             download=True)

    correct = 0
    for sample, label in test_set:
        matmul = ops.matmul_2d(weights_list, [[s.item()] for s in sample])
        forward = [matmul[i][0] + bias_list[i] for i in range(len(bias_list))]
        pred = forward.index(max(forward))
        if pred == label:
            correct += 1

    return correct / len(test_set) * 100


if __name__ == '__main__':
    acc = test_accuracy('./../../../Data/mnist_weights.txt', './../../../Data/mnist_bias.txt')
    print(f'Network accuracy: {acc:.2f}%')
