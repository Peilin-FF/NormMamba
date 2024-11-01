import torch

# 获取torch版本
torch_version = torch.__version__

# 获取CUDA是否可用以及CUDA版本
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else "CUDA not available"

print(torch_version)
print(cuda_version)
