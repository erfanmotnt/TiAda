import torch
from optimizers import TiAda
from torch.optim import Adagrad
torch.set_default_dtype(torch.float64)


number_of_iterations = 8000
alpha = 0.6
beta = 0.4
L = 2
is_TiAda = False
initial_x = 1
grad_noise_x = 0
initial_y = 0.01
grad_noise_y = 0

ratio = 2
learning_rate_y = 0.2 

learning_rate_x = learning_rate_y / ratio
quadratic_fun = lambda x, y: -1/2 * (y ** 2) + L * x * y - (L ** 2 / 2) * (x ** 2)


init_x = torch.Tensor([initial_x])
x = torch.nn.parameter.Parameter(init_x.clone())
init_y = torch.Tensor([initial_y])
y = torch.nn.parameter.Parameter(init_y.clone())

print("initial models")
if is_TiAda:
    optim_y = TiAda([y], lr=learning_rate_y, alpha=beta)
    optim_x = TiAda([x], opponent_optim=optim_y, lr=learning_rate_x, alpha=alpha)
else :
    optim_x = Adagrad([x], lr=learning_rate_x)
    optim_y = Adagrad([y], lr=learning_rate_y)
print("end initialization")

i = 0
while i < number_of_iterations:
    optim_x.zero_grad()
    optim_y.zero_grad()
    l = quadratic_fun(x, y)
    l.backward()

    i += 2
    x_grad_norm = torch.norm(x.grad).item()

    y.grad = -y.grad + grad_noise_y * torch.randn(1)
    x.grad += grad_noise_x * torch.randn(1)
    optim_y.step()
    optim_x.step()

    if i%1000 == 0:
        print('x_grad', f'step: {i}', f'value: {x_grad_norm}')
        print('x', f'step: {i}', f'value: {x.item()}')
        print('y', f'step: {i}', f'value: {y.item()}')
    if x_grad_norm > 1e4:
        break
