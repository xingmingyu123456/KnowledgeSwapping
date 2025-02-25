
import torch
print(torch.cuda.is_available())

import torch
print(torch.version.cuda)

import os
import torch

print("CUDA_HOME:", os.environ.get('CUDA_HOME'))
print("torch.cuda.is_available():", torch.cuda.is_available())