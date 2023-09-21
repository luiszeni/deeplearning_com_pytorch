import torch

tensor_2D = torch.rand((2,4))


print("Concatena linhas:\n", torch.cat((tensor_2D,  -torch.ones(3,4)), dim=0))
print("Concatena colunas:\n", torch.cat((tensor_2D, -torch.ones(2,2)), dim=1))


empilhado = torch.stack((tensor_2D, -torch.ones(2,4)))
print(f"Empilhamento, shape {empilhado.shape}:\n", empilhado)
print()


