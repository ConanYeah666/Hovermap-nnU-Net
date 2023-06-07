
import torch
import numpy as np
from torch import nn
size = 3
h_range = torch.arange(
    -size // 2 + 1,
    size // 2 + 1,
    dtype=torch.float32,
    device="cuda",
    requires_grad=False,
)
v_range = torch.arange(
    -size // 2 + 1,
    size // 2 + 1,
    dtype=torch.float32,
    device="cuda",
    requires_grad=False,
)
print(h_range)
print(v_range)
h, v = torch.meshgrid(h_range, v_range)
print(h)
print(v)
kernel_h = h / (h * h + v * v + 1.0e-15)
kernel_v = v / (h * h + v * v + 1.0e-15)
print(kernel_h)
print(kernel_v)
n=3
if n % 2 == 0:
    raise ValueError("The kernel size must be an odd number.")

base_filter = np.multiply(1/4 , np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]))

sobel_x = np.stack([
    base_filter,
    2 * base_filter,
    base_filter
])

sobel_y = np.stack([
    base_filter.T,
    2 * base_filter.T,
    base_filter.T
])

sobel_z = np.multiply(1/4 , np.array([
    [[-1, -2, -1],
     [-2, -4, -2],
     [-1, -2, -1]],
    [[ 0,  0,  0],
     [ 0,  0,  0],
     [ 0,  0,  0]],
    [[ 1,  2,  1],
     [ 2,  4,  2],
     [ 1,  2,  1]]
]))


print(torch.tensor(sobel_x, device='cuda:0', dtype=float32), torch.tensor(sobel_y, device='cuda:0',dtype=float32), torch.tensor(sobel_z, device='cuda:0',dtype=float32))