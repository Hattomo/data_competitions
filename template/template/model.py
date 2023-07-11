# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class HubertCTC(nn.Module):
    def __init__(self, hubertctc,):
        super(HubertCTC, self).__init__()
        self.hubertctc = hubertctc
        self.lstm = nn.LSTM()

    def forward(self, x):
        self.hubertctc(x)
        return 
