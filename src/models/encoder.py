import torch
import torch.nn as nn

import copy

from models.utils import ReluNet


class IOPairSetEncoder(nn.Module):
    """DeepSet encoder for input-output pairs (N per batch)"""
    def __init__(self,
                 phi=ReluNet(2, 32, 32),
                 rho_0=ReluNet(32, 32, 16),
                 rho_1=ReluNet(16, 8, 3)):
        super().__init__()
        # Shared encoder phi
        self.phi = phi
        # Output processor rho
        self.rho_0 = rho_0
        self.rho_1 = rho_1  # predict mean
        self.rho_2 = copy.deepcopy(rho_1)  # predict logvar

    def forward(self, x):
        # x: (B*, N, 2)
        x_phi = self.phi(x)                 # (B*, N, H)
        # sum over the N-axis (sum set elements' representations)
        x_sum = x_phi.mean(dim=-2)            # (B*, H)
        common = self.rho_0(x_sum)            # (B*, H)
        mu = self.rho_1(common)               # (B*, H)
        logvar = self.rho_2(common)
        return mu, logvar
