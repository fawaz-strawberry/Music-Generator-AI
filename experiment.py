import torch
import torch.nn as nn


input = torch.randn(128, 100, 1, 1)

Z_DIM = 100
FEATURES_GEN = 64

m = nn.ConvTranspose2d(Z_DIM, FEATURES_GEN * 32, (6, 4), 1, 0)
output = m(input)
print("Single Layer")
print(output.shape)

m2 = nn.ConvTranspose2d(FEATURES_GEN * 32, FEATURES_GEN * 16, (5, 2), 2, 1)
output = m2(output)
print("Double Layer")
print(output.shape)

m3 = nn.ConvTranspose2d(FEATURES_GEN * 16, FEATURES_GEN * 8, (5, 3), 2, 1)
output = m3(output)
print("Triple Layer")
print(output.shape)

m4 = nn.ConvTranspose2d(FEATURES_GEN * 8, FEATURES_GEN * 4, (4, 2), 2, 1)
output = m4(output)
print("Quadruple Layer")
print(output.shape)

m5 = nn.ConvTranspose2d(FEATURES_GEN * 4, FEATURES_GEN * 2, (4, 3), 2, 1)
output = m5(output)
print("Pentatic Layer")
print(output.shape)

m6 = nn.ConvTranspose2d(FEATURES_GEN * 2, FEATURES_GEN, (3, 3), 2, 1)
output = m6(output)
print("Hexagonal Layer")
print(output.shape)

m7 = nn.ConvTranspose2d(FEATURES_GEN, 3, (4, 3), 2, 1)
output = m7(output)
print("Hexagonal Layer")
print(output.shape)
