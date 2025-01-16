import torch
import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.utils.weight_norm(nn.Linear(input_dim, 128))
        self.fc2 = nn.utils.weight_norm(nn.Linear(128, 128))
        self.fc3 = nn.utils.weight_norm(nn.Linear(128, 128))
        self.fc4 = nn.utils.weight_norm(nn.Linear(128, output_dim))
        self.dropout = nn.Dropout(0.05)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
