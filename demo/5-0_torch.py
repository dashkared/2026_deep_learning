import torch

x = torch.randn(32, 128, requires_grad=False)
w = torch.randn(128, 10, requires_grad=True)
y = x @ w
loss = y.square().mean()
loss.backward()

print(x.grad)
print(w.grad)
