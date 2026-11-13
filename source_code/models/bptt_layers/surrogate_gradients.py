import torch

class GradSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        value = (ctx.saved_tensors[0] * 4).sigmoid_()
        return (1 - value) * value * 4 * grad_output
