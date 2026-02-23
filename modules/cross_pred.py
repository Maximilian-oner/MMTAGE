"""
 Copyright (c) 2026, Bioinformatics Institute.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torch import nn
from torch.nn import functional as F


class EmbtoGeneDecoder(nn.Module):
    def __init__(self, gene_num=5000):
        super(EmbtoGeneDecoder, self).__init__()
        self.fc1 = nn.Linear(64, 1024)
        self.fc2 = nn.Linear(1024, gene_num)
        self.relu = nn.ReLU()

    def forward(self, emb, gex_targets):
        x = self.relu(self.fc1(emb))
        pred = self.fc2(x)

        cross_loss = F.mse_loss(pred, gex_targets)

        return cross_loss

