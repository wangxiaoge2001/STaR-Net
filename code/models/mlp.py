import torch.nn as nn
from models import register

@register('mlp')
class MLP(nn.Module):

    def __init__(self, inChannels, outChannels, hidden_list, act='gelu'):
        super().__init__()
        if act.lower() == 'relu':
            self.act = nn.ReLU()
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        else:
            assert False, f'activation {act} is not supported'
        layers = []
        lastv = inChannels
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, outChannels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
