import torch

tensor_nD = torch.ones((1,2,4,2), device='cuda', dtype=torch.int)

print("Propriedades dos tensors:")
print("- Formato:", tensor_nD.shape)
print("- Tipo de dado:", tensor_nD.dtype)
print("- Dispositivo:", tensor_nD.device)
print("- Número de dimenssões do tensor:", tensor_nD.ndim)
print()
