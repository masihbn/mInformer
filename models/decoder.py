import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayerNN(nn.Module):
    def __init__(self, layers_dimensions, seq_length, device=None):
        super(DecoderLayerNN, self).__init__()

        self.nn_layers = []
        for i in range(0, len(layers_dimensions) - 1):
            self.nn_layers.append(nn.Linear(layers_dimensions[i], layers_dimensions[i + 1]))
            self.nn_layers.append(nn.ReLU())
            self.nn_layers.append(nn.Dropout(p=0.5))

        self.output_dimension = layers_dimensions[-1] * seq_length
        self.output_layer = nn.Linear(self.output_dimension, 1)

    def forward(self, x, cross=None, x_mask=None, cross_mask=None):
        for layer in self.nn_layers:
            layer.cuda()
            x = layer(x)

        x = x.view(-1, self.output_dimension)
        out = self.output_layer(x)

        return out


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross=None, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
