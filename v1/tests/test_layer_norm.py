import torch

from model.layer_norm import LayerNormalization

layer = LayerNormalization(32)

x = torch.randn(
    2,
    8,
    32
)

output = layer(x)

print(output.shape)