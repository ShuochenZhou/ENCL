import torch
import torch.nn as nn

class Masking:
    def __init__(self, net, density, sparse_init, device):
        self.density = density
        self.masks = {}
        self.net = net
        self.device = device
        for name, parameter in net.named_parameters():
            if parameter.numel() and parameter.dim():
                self.masks[name] = torch.zeros_like(parameter, dtype=torch.float32, requires_grad=False).to(device)
        for name in list(self.masks):
            if "bias" in name:
                self.masks.pop(name)
        for module_name, module in net.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.masks.pop(module_name + ".weight", None)
        if sparse_init != "uniform":
            raise ValueError("This experiment uses only uniform mask initialization")
        for name, parameter in net.named_parameters():
            if name in self.masks:
                self.masks[name][:] = (torch.rand(parameter.shape) < density).float().data.to(device)
        self._apply_mask()

    def step(self, epoch):
        self._apply_mask()

    def _apply_mask(self):
        for name, parameter in self.net.named_parameters():
            if name in self.masks:
                parameter.data = parameter.data * self.masks[name]
