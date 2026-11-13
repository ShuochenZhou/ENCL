import torch
import torch.nn as nn
import torch.nn.functional as F
from .bptt_layers.spiking_layers import BPTTLIF, DropoutLIF
from .bptt_layers.surrogate_gradients import GradSigmoid


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-4):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias
        )
        self.normalization = nn.Parameter(
            torch.ones(self.out_channels, 1, 1, 1)
        )
        self.eps = eps

    def get_weight(self):
        support = self.weight.ne(0)
        active = support.sum(dim=[1, 2, 3], keepdim=True).clamp_min(1)
        active_f = active.to(self.weight.dtype)
        mean = (
            (self.weight * support).sum(dim=[1, 2, 3], keepdim=True)
            / active_f
        )
        centered = (self.weight - mean) * support
        variance = (
            centered.square().sum(dim=[1, 2, 3], keepdim=True) / active_f
        )
        return (
            centered / ((variance * active_f + self.eps) ** 0.5)
            * self.normalization
        )

    def forward(self, x):
        return F.conv2d(
            x, self.get_weight(), self.bias, self.stride, self.padding,
            self.dilation, self.groups
        )


class VGG(nn.Module):
    def __init__(
        self,
        n_inputs=3,
        labels=None,
        activation=GradSigmoid,
        dropout=0.0,
        feature_only=True,
    ):
        super().__init__()
        self.conv1 = Conv2d(n_inputs, 64, 3, 1, 1, bias=False)
        self.lif1 = BPTTLIF(activation=activation, leak=0.0)
        self.dropout1 = DropoutLIF(dropout)
        self.conv2 = Conv2d(64, 128, 3, 1, 1, bias=False)
        self.lif2 = BPTTLIF(activation=activation, leak=0.0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout2 = DropoutLIF(dropout)
        self.conv3 = Conv2d(128, 256, 3, 1, 1, bias=False)
        self.lif3 = BPTTLIF(activation=activation, leak=0.0)
        self.dropout3 = DropoutLIF(dropout)
        self.conv4 = Conv2d(256, 256, 3, 1, 1, bias=False)
        self.lif4 = BPTTLIF(activation=activation, leak=0.0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout4 = DropoutLIF(dropout)
        self.conv8 = Conv2d(256, 512, 3, 1, 1, bias=False)
        self.lif8 = BPTTLIF(activation=activation, leak=0.0)
        self.dropout5 = DropoutLIF(dropout)
        self.conv9 = Conv2d(512, 512, 3, 1, 1, bias=False)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.lif9 = BPTTLIF(activation=activation, leak=0.0)
        self.dropout6 = DropoutLIF(dropout)
        self.conv11 = Conv2d(512, 512, 3, 1, 1, bias=False)
        self.lif11 = BPTTLIF(activation=activation, leak=0.0)
        self.dropout7 = DropoutLIF(dropout)
        self.conv12 = Conv2d(512, 512, 3, 1, 1, bias=False)
        self.lif12 = BPTTLIF(activation=activation, leak=0.0)
        self.dropout8 = DropoutLIF(dropout)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512
        self.linear16 = None
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def reset_states(self):
        for layer in (
            self.lif1, self.lif2, self.lif3, self.lif4,
            self.lif8, self.lif9, self.lif11, self.lif12,
            self.dropout1, self.dropout2, self.dropout3, self.dropout4,
            self.dropout5, self.dropout6, self.dropout7, self.dropout8,
        ):
            layer.reset_state()

    def forward_features(self, x, target=None):
        spike_rates = []
        x = self.lif1(self.conv1(x))
        spike_rates.append(float(x.detach().ne(0).float().mean().item()))
        x = self.dropout1(x)
        x = self.lif2(self.conv2(x))
        spike_rates.append(float(x.detach().ne(0).float().mean().item()))
        x = self.pool1(self.dropout2(x))
        x = self.lif3(self.conv3(x))
        spike_rates.append(float(x.detach().ne(0).float().mean().item()))
        x = self.dropout3(x)
        x = self.lif4(self.conv4(x))
        spike_rates.append(float(x.detach().ne(0).float().mean().item()))
        x = self.pool2(self.dropout4(x))
        x = self.lif8(self.conv8(x))
        spike_rates.append(float(x.detach().ne(0).float().mean().item()))
        x = self.dropout5(x)
        x = self.lif9(self.conv9(x))
        spike_rates.append(float(x.detach().ne(0).float().mean().item()))
        x = self.pool4(self.dropout6(x))
        x = self.lif11(self.conv11(x))
        spike_rates.append(float(x.detach().ne(0).float().mean().item()))
        x = self.dropout7(x)
        x = self.lif12(self.conv12(x))
        spike_rates.append(float(x.detach().ne(0).float().mean().item()))
        x = self.dropout8(x)
        x = self.globalpool(x).flatten(1)
        self._last_spike_rates = spike_rates
        self._last_feature_nonzero = float(x.detach().ne(0).float().mean().item())
        self._last_feature_abs_mean = float(x.detach().abs().mean().item())
        return x

    def forward(self, x, target=None):
        return self.forward_features(x, target)

def vgg_bptt(args, device):
    return VGG()
