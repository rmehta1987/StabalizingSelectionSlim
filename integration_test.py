import torch

def f_x(x,s):

    N = 100.0
    
    u = .1
    i = 1.0
    scaled_pop_mut = 4*N*u
    numerator = (1 - torch.exp(-2*s*(1-x))) * (torch.pow(x,i))*torch.pow((1-x),(N-i))
    denomiator = x*(1-x)
    
    return (scaled_pop_mut*numerator)/denomiator


def f_x2(x,s):
    
    return torch.pow(x,s)

points = torch.linspace(0.001, 2, 10)
points2 = torch.linspace(1 , 2, 2)
s = torch.Tensor([2])
s.requires_grad=True
values = f_x2(points2, s)
result = torch.trapz(y=values)

# pytorch trapz formula
# sum_{i=1}^_{n-1} (x_i - x_{i-1})/2 * (y_i + y_{i-1})
test_case = 0.0
for i in range(1,points2.shape[0]):
    test_case = test_case + (points2[i]-points2[i-1])/2 * (values[i]+values[i-1])

    
loss = torch.autograd.grad(result, s)

# assuming integration domain ins [1, 2]
top = torch.pow(-2,s+1) + torch.pow(2,(s+1))*(s+1)*torch.log(s)+1
bot = (s+1)**2
expected = top/bot

print(loss)
print(expected)