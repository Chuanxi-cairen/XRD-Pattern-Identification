import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ================== 1. Dataset and model definitions ==================
class XRDDataset(Dataset):
    def __init__(self, sample, label):
        self.samples = torch.tensor(np.array(sample), dtype=torch.float32).permute(0, 2, 1)
        self.labels = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class CustomDropout(nn.Module):
    def __init__(self, rate):
        super(CustomDropout, self).__init__()
        self.rate = rate

    def forward(self, x):
        return F.dropout(x, self.rate, training=self.training)


class XRDNet(nn.Module):
    def __init__(self, n_phases, best_params):
        super(XRDNet, self).__init__()
        self.batch_size = best_params['batch_size']
        self.n_conv_layers = best_params['n_conv_layers']
        self.lr = best_params['lr']
        self.dropout_rate = best_params['dropout_rate']
        self.dense_units = best_params['dense_units']

        self.conv_layers = nn.Sequential()
        in_channels = 1
        for i in range(self.n_conv_layers):
            out_channels = best_params[f'out_ch_{i}']
            kernel_size = best_params[f'ksize_{i}']
            padding = best_params[f'pad_{i}']
            pool_size = best_params[f'pool_{i}']

            self.conv_layers.add_module(f'conv_{i}', nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU())
            self.conv_layers.add_module(f'pool_{i}', nn.MaxPool1d(kernel_size=pool_size, stride=2))
            in_channels = out_channels

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self._get_flatten_size(), self.dense_units[0]),
            nn.BatchNorm1d(self.dense_units[0]),
            nn.ReLU(),
            CustomDropout(self.dropout_rate),
            nn.Linear(self.dense_units[0], self.dense_units[1]),
            nn.BatchNorm1d(self.dense_units[1]),
            nn.ReLU(),
            CustomDropout(self.dropout_rate),
            nn.Linear(self.dense_units[1], n_phases)
        )

    def _get_flatten_size(self):
        dummy = torch.zeros(1, 1, 4501)
        return torch.flatten(self.conv_layers(dummy), start_dim=1).size(1)

    def get_feature(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        return x

    def get_deep_feature(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i == 6:  # ReLU after fc2
                return x

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x