import torch

tensor_escalar = torch.tensor(42)
tensor_vetor   = torch.tensor([-1,0,0,1], dtype=torch.float)
tensor_2D      = torch.rand((2,4))


print("Operações básicas elemento a elemento")
print("- Soma:\n", tensor_2D + tensor_2D)
print("- Subtração:\n", tensor_2D - tensor_2D)
print("- Multiplicação:\n", tensor_2D * tensor_2D)
print("- Divisão:\n", tensor_2D / tensor_2D)
print()


print("Broadcasting:")
resultado = tensor_2D + tensor_escalar
print("Soma de matriz com escalar:\n", resultado)
print()


print("Funções matematicas comuns:")
exponencial = torch.exp(tensor_vetor)
raiz_quadrada = torch.sqrt(tensor_vetor)
print("- Exponecial:\n", exponencial)
print("- Raiz Quadrada:\n", raiz_quadrada)
print()


print("Indexação e Fatiamento:")
print("- Elemento no indice [1, 1]:", tensor_2D[1, 1])
print("- Primeira linha:", tensor_2D[0])
print("- Segunda coluna", tensor_2D[:, 1])
print("- Seleciona a partir da primeira linha e coluna:\n", tensor_2D[:1, :3])
print()


tensor_2D_reformatado = tensor_2D.view(4, 2)
print("Reformatado com .view():\n", tensor_2D_reformatado)
print(f"shape antigo:{tensor_2D.shape}, shape novo:{tensor_2D_reformatado.shape}")
print()

tensor_2D_reformatado = tensor_2D.reshape(8, 1)
print("Reformatado com .reshape():\n", tensor_2D_reformatado)
print(f"shape antigo:{tensor_2D.shape}, shape novo:{tensor_2D_reformatado.shape}")
print()


print("Multiplicação matricial:\n", torch.matmul(tensor_2D, tensor_vetor))
print()
