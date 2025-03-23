import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """Helper function to get images from directories"""
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    images = [(i, os.path.join(path, image))
              for i, path in zip(labels, paths)
              for image in sampler(os.listdir(path))]

    if shuffle:
        random.shuffle(images)

    return images


class ConvBlock(nn.Module):
    """Convolutional block with optional max pooling and normalization"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_type='batch_norm',
                 max_pool=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm_type = norm_type

        if norm_type == 'batch_norm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'layer_norm':
            self.norm = nn.LayerNorm([out_channels, 28, 28])  # Adjust spatial dims as needed
        else:
            self.norm = None

        self.activation = nn.ReLU()
        self.max_pool = max_pool

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)
        if self.max_pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x


def mse_loss(pred, label):
    """Mean Squared Error Loss"""
    return F.mse_loss(pred.view(-1), label.view(-1))


def cross_entropy_loss(pred, label):
    """Cross-entropy loss with softmax"""
    return F.cross_entropy(pred, label)
