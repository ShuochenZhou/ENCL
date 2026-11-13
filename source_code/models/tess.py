FACTORS = [0.2, 0.75, -1, 1]

import torch.nn as nn
from .tess_layers import (
    # Standard TESS convolution layer.
    Conv2dSTDP,
    DropoutLIF,
    LocalLearningSignalGenerationLayer,
)

class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, pool_size, adaptive, factors):
        super().__init__()
        if adaptive:
            self.pool_layer = nn.AdaptiveAvgPool2d
            self.pool = self.pool_layer(1)
        else:
            self.pool_layer = nn.MaxPool2d
            self.pool = nn.Identity() if pool_size == 1 else self.pool_layer(pool_size)
        # Conv2dSTDP follows the standard TESS eligibility-trace convention.
        self.conv = Conv2dSTDP(
            n_inputs, n_outputs, 3, 1, 1, bias=False, leak=0, factors=factors
        )
        self.dropout = DropoutLIF(0.0)
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_out", nonlinearity="relu"
        )

    def reset_state(self):
        self.conv.reset_state()
        self.dropout.reset_state()

    def forward(self, input):
        return self.pool(self.dropout(self.conv(input)))

class VGGConvModel(nn.Module):
    def __init__(self, labels, local_lr):
        super().__init__()
        factors = FACTORS
        channels = [3, 64, 128, 256, 256, 512, 512, 512, 512]
        pool_sizes = [1, 2, 1, 2, 1, 2, 1, 1]
        pooling_sizes = [4, 2, 2, 2, 2, 2, 2, 1]
        hidden = [4 * 4 * 64, 2 * 2 * 128, 2 * 2 * 256, 2 * 2 * 256,
                  2 * 2 * 512, 2 * 2 * 512, 2 * 2 * 512, 512]
        for index in range(8):
            block = ConvBlock(
                channels[index], channels[index + 1], pool_sizes[index],
                index == 7, factors
            )
            layer = LocalLearningSignalGenerationLayer(
                block, local_lr, labels, hidden[index], pooling_sizes[index]
            )
            setattr(self, f"conv_block{index + 1}", layer)
        self.feature_dim = 512
        self.linear = None
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            # Reset state carried by the standard TESS Conv2dSTDP layer.
            if isinstance(module, Conv2dSTDP):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def _conv_blocks(self):
        return [getattr(self, f"conv_block{index}") for index in range(1, 9)]

    def clear_local_gradients(self):
        for block in self._conv_blocks():
            block.clear_gradients()

    def apply_local_plain_sgd(self):
        for block in self._conv_blocks():
            block.apply_plain_sgd()

    def reset_states(self):
        for block in self._conv_blocks():
            block.block.reset_state()

    def forward_features(self, x, target=None):
        batch_size = x.shape[0]
        for block in self._conv_blocks():
            x = block(x, target)
        return x.view(batch_size, -1)

    def forward(self, x, target=None):
        return self.forward_features(x, target)

def vgg_tess(args, device):
    return VGGConvModel(args.num_classes, args.lr_conv)
