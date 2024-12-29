import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.distributed as dist
from typing import List, Optional, Tuple, Callable, Union

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor,0)
        self.std = torch.std(tensor,0)

    def norm(self, tensor):
        return (tensor - self.mean.to(tensor.device)) / self.std.to(tensor.device)

    def denorm(self, normed_tensor):
        return normed_tensor * self.std.to(normed_tensor.device) + self.mean.to(normed_tensor.device)

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class EarlyStopping(object):
    """Early stopping.

    Parameters
    ----------
    patience : int = 10
        Patience for early stopping.

    """

    best_losses = None
    counter = 0

    def __init__(self, patience: int = 10):
        self.patience = patience

    def __call__(self, losses):
        if self.best_losses is None:
            self.best_losses = losses
            self.counter = 0

        elif any(
            loss <= best_loss
            for loss, best_loss in zip(losses, self.best_losses)
        ):
            if all(
                loss <= best_loss
                for loss, best_loss in zip(losses, self.best_losses)
            ):

                self.best_losses = [
                    min(loss, best_loss)
                    for loss, best_loss in zip(losses, self.best_losses)
                ]
                self.counter = 0

        else:
            self.counter += 1
            if self.counter == self.patience:
                return True

        return False

class gate_nn(nn.Module):
    '''
    Gate function for the attention pooling layer
    '''
    def __init__(self, in_features: int,hidden_features: int,out_dim:int,activation: Callable = nn.LeakyReLU()):
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        self.act = activation
        self.out = nn.Linear(hidden_features, out_dim)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.out(x)
        return x

class feat_nn(nn.Module):
    '''
    Feature function for the attention pooling layer
    '''
    def __init__(self, in_features: int, hidden_features: int, out_features: int,activation: Callable = nn.LeakyReLU()):
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        self.act = activation
        self.out = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.out(x)
        return x