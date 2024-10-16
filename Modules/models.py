import os
from abc import ABC

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class GenericNN(nn.Module):

    def __init__(self):

        super().__init__()
        self.loaded = False
        self.extension = ""
        self.optimizer = None
        self.device = "cpu"

    def reset_weights(self):
        """ Apply the function to reset weights """
        self.apply(self.__reset_model_weights__)

    @staticmethod
    def __reset_model_weights__(m):
        """ Reset the models parameters. """
        try:
            m.reset_parameters()
        except AttributeError as e:
            pass

    def to(self, device, *args, **kwargs):
        self.device = device
        super().to(device, *args, **kwargs)


class Predictor(GenericNN):

    def __init__(self, input_dim, predictor_dim, output_dim=1, output_probs="log_softmax", checkpointing=True):
        super(Predictor, self).__init__()
        dim = input_dim
        self.output_dim = output_dim
        seq = []
        for item in list(predictor_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, output_dim)]
        if output_probs == "log_softmax":
            seq.append(nn.LogSoftmax(dim=1))
        elif output_probs == "sigmoid":
            seq.append(nn.Sigmoid())
        elif output_probs == "softmax":
            seq.append(nn.Softmax(dim=1))
        else:
            seq.append(nn.LeakyReLU())

        self.checkpointing = checkpointing
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        """
        Compute the forward pass of the network
        Args:
            x: input data

        Returns:

        """
        if self.checkpointing:
            # necessary for checkpointing :
            # see https://discuss.pytorch.org/t/use-of-torch-utils-checkpoint-checkpoint-causes-simple-model-to-diverge/116271
            if not x.requires_grad:
                x.requires_grad = True
            data = checkpoint.checkpoint(self.seq, x)
        else:
            data = self.seq(x)
        return data


class Generator(GenericNN):
    """
    Generator model class
    """
    def __init__(self, input_dim, generator_dim, data_dim, checkpointing=True):
        super(Generator, self).__init__()
        dim = input_dim
        seq = []
        for item in list(generator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item
        seq.append(nn.Linear(dim, data_dim))
        seq.append(nn.LeakyReLU(0.2))

        self.seq = nn.Sequential(*seq)
        self.checkpointing = checkpointing

    def forward(self, x, controls=None, noise=None):
        """
        Compute the forward pass of the generator
        Args:
            x: input data
            controls: sensitive groups that goes along with the input depending on your objective
            noise: Noise that goes along with the input depending on your objective

        Returns:
            Modified data
        """
        if controls is not None:
            x = torch.cat((x, controls), 1)
        if noise is not None:
            x = torch.cat((x, noise), 1)
        if self.checkpointing:
            # necessary for checkpointing :
            # see https://discuss.pytorch.org/t/use-of-torch-utils-checkpoint-checkpoint-causes-simple-model-to-diverge/116271
            if not x.requires_grad:
                x.requires_grad = True
            data = checkpoint.checkpoint(self.seq, x)
        else:
            data = self.seq(x)
        return data

