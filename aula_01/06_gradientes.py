import torch

####
# Gradientes e autograd
####
tensor1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
tensor2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

result = (tensor1 - tensor2)
print("Result tensor:", result)

print("Calculando os gradientes utilizando .backward():")
loss = result.sum()
loss.backward()

print("Gradientes:")
print("Gradientes do tensor1:", tensor1.grad)
print("Gradientes do tensor2:", tensor2.grad)
