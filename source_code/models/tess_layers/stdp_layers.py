import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# The class names and eligibility-trace update follow standard TESS practice.
class STDPConv2dGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups,
                trace_in, trace_out, mem, leak, threshold, factors, scale):
        with torch.no_grad():
            trace_in = factors[1] * trace_in + input
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
            mem = torch.sigmoid(leak) * mem + output
            u_thr = mem - threshold.clamp(min=0.5)
            out = (u_thr > 0).float()
            psi = 0.3 * F.threshold(1.0 - torch.abs(u_thr), 0, 0)
            trace_out_next = factors[0] * trace_out + psi
        ctx.save_for_backward(input, weight, bias, trace_in, trace_out, psi, threshold, factors)
        ctx.in1 = [stride, padding, dilation, groups]
        return out, mem, trace_in, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem, grad_trace_in, grad_trace_out):
        input, weight, bias, trace_in, trace_out, psi, threshold, factors = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.in1
        grad = psi * grad_output
        grad_input = torch.nn.grad.conv2d_input(
            input.size(), weight, grad, stride, padding, dilation, groups
        )
        delta_weight_pre = torch.nn.grad.conv2d_weight(
            trace_in, weight.size(), grad_output * psi, stride, padding, dilation, groups
        )
        delta_weight_post = torch.nn.grad.conv2d_weight(
            input, weight.size(), grad_output * trace_out, stride, padding, dilation, groups
        )
        grad_weight = factors[2] * delta_weight_post + factors[3] * delta_weight_pre
        grad_bias = None if bias is None else grad.sum(dim=(0, 2, 3))
        return (grad_input, grad_weight, grad_bias, None, None, None, None,
                None, None, None, None, None, None, None)

# Standard TESS layer name and learning rule.
class Conv2dSTDP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros",
                 threshold=0.6, leak=2.0, grad_tl=False, factors=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode
        )
        self.u = None
        self.state_reset = None
        self.trace_in = None
        self.trace_out = None
        self.leak = nn.Parameter(torch.tensor(leak), requires_grad=grad_tl)
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=grad_tl)
        self.reset_mechanism = "soft"
        self.accumulate = False
        self.normalization = nn.Parameter(
            torch.ones(self.out_channels, 1, 1, 1)
        )
        self.wn = False
        self.avoid_wn = False
        # Standard TESS eligibility-trace autograd rule.
        self.activation = STDPConv2dGrad.apply
        self.eps = 1e-4
        self.register_buffer("factors", torch.tensor(factors))
        self.register_buffer("conv_scaling_factor", torch.ones([1]))
        self.weight = Parameter(self.weight.data.clone())
        self.register_buffer("weight_integer", torch.zeros_like(self.weight.data))
        self.register_buffer("w_min", torch.zeros(1))
        self.register_buffer("w_max", torch.zeros(1))

    def get_weight(self):
        with torch.no_grad():
            mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
            fixed_var = getattr(self, "hat_fixed_var", None)
            var = (fixed_var if fixed_var is not None else
                   (torch.max(self.weight) - torch.mean(self.weight)).detach())
        return (self.weight - mean.detach()) / var

    def reset_state(self):
        if self.u is not None:
            with torch.no_grad():
                self.u = self.u.mul(0).detach()
                self.trace_in = self.trace_in.mul(0).detach()
                self.trace_out = self.trace_out.mul(0).detach()

    def _init_states(self, x):
        batch_size = x.size(0)
        if self.u is None or self.u.shape[0] != batch_size:
            with torch.no_grad():
                a = F.conv2d(
                    x, self.weight, None, self.stride, self.padding,
                    self.dilation, self.groups
                )
                self.u = torch.zeros_like(a).to(x.device)
                self.trace_out = torch.zeros_like(a).to(x.device)
        if self.trace_in is None or self.trace_in.shape[0] != batch_size:
            self.trace_in = torch.zeros_like(x).to(x.device)

    def forward(self, input):
        self._init_states(input)
        out, u, trace_in, trace_out = self.activation(
            input, self.get_weight(), self.bias, self.stride, self.padding,
            self.dilation, self.groups, self.trace_in, self.trace_out, self.u,
            self.leak, self.threshold, self.factors, self.conv_scaling_factor
        )
        self.u = u.detach()
        self.trace_out = trace_out.detach()
        self.trace_in = trace_in.detach()
        self.u = self.u - torch.round(
            self.threshold.clamp(min=0.5) / self.conv_scaling_factor
        ) * out.detach()
        return out
