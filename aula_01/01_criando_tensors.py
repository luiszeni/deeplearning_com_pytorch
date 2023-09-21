import torch

tensor_escalar = torch.tensor(42)
print('Tensor representando um escalar:\n', tensor_escalar)

tensor_vetor = torch.tensor([-1,0,0,1], dtype=torch.float)
print('Tensor representando um vetor:\n', tensor_vetor)
print()

tensor_2D = torch.rand((2,4))
print('Tensor representando uma matriz 2d:\n', tensor_2D)
print()

tensor_3D = torch.zeros((2,4,2), dtype=torch.bool)
print('Tensor representando um tensor 3D:\n', tensor_3D)
print()

tensor_nD = torch.ones((1,2,4,2), device='cuda', dtype=torch.int)
print('Tensor representando um tensor n-dimenssional:\n', tensor_nD)
print()

tensor_vetor_em_int = tensor_vetor.to(torch.int)
print(f'Conversão, de {tensor_vetor.dtype} para {tensor_vetor_em_int.dtype}')


tensor_vetor_em_int = tensor_vetor.int()
print(f'Conversão, de {tensor_vetor.dtype} para {tensor_vetor_em_int.dtype}')
print()

tensor_np = tensor_vetor.numpy()
print("Numpy Array:\n", tensor_np, "\nTipo:", type(tensor_np))
print("Tensor from numpy:\n", torch.from_numpy(tensor_np))
print()

print('Tensor da cpu p/ gpu:\n', tensor_vetor.cuda())
print('Tensor da gpu p/ cpu:\n', tensor_nD.cpu())

print('Tensor da cpu p/ gpu:\n', tensor_vetor.to('cuda'))
print('Tensor da gpu p/ cpu:\n', tensor_nD.to('cpu'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Dispositivo:', device, '-> Tensor enviado:\n', tensor_3D.to(device))
