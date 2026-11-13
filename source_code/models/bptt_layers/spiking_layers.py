import copy
import torch
import torch.nn as nn

class BPTTLIF(nn.Module):
    def __init__(self, threshold=0.6, leak=2.0, grad_tl=False, activation=None):
        super().__init__()
        self.u = None
        self.state_reset = None
        self.leak = nn.Parameter(torch.tensor(leak), requires_grad=grad_tl)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=grad_tl)
        self.activation = activation.apply

    def reset_state(self):
        self.u = copy.deepcopy(self.state_reset)

    def forward(self, x):
        if self.u is None or self.u.shape[0] != x.shape[0]:
            self.u = torch.zeros_like(x).to(x.device)
            self.state_reset = torch.zeros_like(x).to(x.device)
        self.u = torch.sigmoid(self.leak) * self.u + x
        threshold = self.threshold.clamp(min=0.5)
        output = self.activation(self.u - threshold)
        self.u = self.u - threshold * output.detach()
        return output

class DropoutLIF(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.mask = None
        self.ones = torch.ones(1)
        self.dropout = nn.Dropout(p)

    def reset_state(self):
        self.mask = self.dropout(self.ones).trunc()

    def forward(self, x):
        if self.mask.dim() == 1 or self.mask.shape[0] != x.shape[0]:
            self.ones = torch.ones_like(x)
            self.mask = self.dropout(self.ones).trunc()
        return self.mask * x
