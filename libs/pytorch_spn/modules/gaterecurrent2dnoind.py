import torch.nn as nn
from ..functions.gaterecurrent2dnoind import GateRecurrent2dnoindFunction

class GateRecurrent2dnoind(nn.Module):
    """docstring for ."""
    def __init__(self, horizontal_, reverse_):
        super(GateRecurrent2dnoind, self).__init__()
        self.horizontal = horizontal_
        self.reverse = reverse_

    def forward(self, X, G1, G2, G3):
        return GateRecurrent2dnoindFunction(self.horizontal, self.reverse)(X, G1, G2, G3)
