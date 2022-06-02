import torch
from torchquad import MonteCarlo, set_up_backend
import matplotlib.pyplot as plt
set_up_backend("torch", data_type="float32")

def f_x(x):
    '''
        poisson random field
    '''
    N = 1000.0
    s = .0001
    u = .00000001
    i = 1.0
    scaled_pop_mut = 4*N*u
    numerator = (1 - torch.exp(-2*s*(1-x))) * (torch.pow(x,i))*torch.pow((1-x),(N-i))
    denomiator = x*(1-x)


mc = MonteCarlo()

integral_value = mc.integrate(
    f_x,
    dim=1,
    N=10000,
    integration_domain=torch.tensor([[0.0, 1.0]]))

points = torch.linspace(0, 1.0, 1000.0)
plt.plot(points.cpu(), f_x(points).cpu())
plt.xlabel("$x$", fontsize=14)
plt.ylabel("f($x$)", fontsize=14)   
    
    
    