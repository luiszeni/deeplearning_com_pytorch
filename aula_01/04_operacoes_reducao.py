import torch

tensor_2D = torch.rand((2,4))

####
# Operações de redução
####
print("Operações de redução:")
print("- Soma:", tensor_2D.sum())
print("- Média:", tensor_2D.mean())
print("- Desvio Padrão:", tensor_2D.std())
print("- Máximo:", tensor_2D.max())
print("- Mínimo:", tensor_2D.min())
print()


print("Redução ao longo das dimenssões:")
print("Soma na direção das linhas (dim=0):\n", tensor_2D.sum(dim=0))
print("Media na direção das colunas (dim=1):\n", tensor_2D.mean(dim=1))
print()

