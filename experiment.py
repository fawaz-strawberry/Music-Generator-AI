import torch
import torch.nn as nn

input = torch.randn(128, 100, 1, 1)

Z_DIM = 100
FEATURES_GEN = 64

m = nn.ConvTranspose2d(Z_DIM, FEATURES_GEN * 16, 4, 1, 0)
output = m(input)
print("Single Layer")
print(output.shape)
m2 = nn.ConvTranspose2d(FEATURES_GEN * 16, FEATURES_GEN * 8, 4, 2, 1)
output = m2(output)
print("Double Layer")
print(output.shape)



