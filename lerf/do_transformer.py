import torch
import torch.nn as nn
from lerf.lerf_fieldheadnames import LERFFieldHeadNames


class Planner(nn.Module):
    def __init__(self):
        pass
    def forward(self, outputs: dict):
        x = outputs[LERFFieldHeadNames.HASHGRID]
        clip_pass = outputs[LERFFieldHeadNames.CLIP]
        dino_pass = outputs[LERFFieldHeadNames.DINO]
        
        