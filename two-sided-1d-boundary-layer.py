# The program solves the two-sided 1D boundary layer problem numerically with 
# the deep Uzawa-Lagrange multiplier approach for PINNs and deep Ritz method.

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Fully connected feed forward neural network with width and depth
class FFNN(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = [nn.Linear(1, width), nn.SiLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# Evaluate model at x (list)
def evaluate(model, x):
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).view(-1, 1)
    return model(x_tensor)

# Compute Dirichlet functional
def dirichlet(epsilon, model, x):
    x.requires_grad_()
    u = model(x)
    u_grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return 0.5 * epsilon * u_grad**2 + 0.5 * u**2 - u

# Compute PINN loss
def pinn(epsilon, model, x):
    x.requires_grad_()
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return (-epsilon * u_xx + u - 1)**2

# Computes Mean Squared Error between model output and the exact solution
def L2_distance(epsilon, model, x):
    u_exact = exact_solution(epsilon, x)
    with torch.no_grad():
        u = model(x)
        return torch.mean((u - u_exact)**2)

# Compute boundary loss
def boundary(model):
    u_0, u_1 = evaluate(model, [0, 1])
    # Raw boundary penalty without multiplication by gamma
    return 0.5 * (u_0**2 + u_1**2)

# Compute Lagrange multiplier term
def lagrange(model, lambda_0, lambda_1):
    u_0, u_1 = evaluate(model, [0, 1])
    return lambda_0 * u_0 + lambda_1 * u_1

# Computes the exact solution (both Tensor and Numpy inputs)
def exact_solution(epsilon, x):
    c = 1.0 / np.sqrt(epsilon)
    
    if torch.is_tensor(x):
        exp_fn = torch.exp
        c = torch.tensor(c, device=x.device, dtype=x.dtype)
    else:
        exp_fn = np.exp
        
    numerator = exp_fn(c * (1 - x)) + exp_fn(c * x)
    denominator = exp_fn(c) + 1
    
    return 1 - numerator / denominator

# Plots model against the exact solution
def plot_solution(model, k, epsilon):
    with torch.no_grad():
        x_plot = np.linspace(0, 1)
        u_plot = evaluate(model, x_plot).detach().numpy()
        u_true = exact_solution(epsilon, x_plot)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x_plot, u_true, label='Exact Solution', color='red', linestyle='--', alpha=0.7)
    plt.plot(x_plot, u_plot, label=f'NN Iteration {k}', color='blue')
    
    plt.xlim(0, 1)
    # Relax ylim slightly to see the curves clearly if they overshoot
    plt.ylim(-0.2, 1.2) 
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'Solution at Uzawa Iteration {k} ($\\epsilon={epsilon}$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_histories(histories):
    """
    Plots the Dirichlet and Boundary energy history over multiple runs.
    Accepts either a single history dict or a list of history dicts.
    """
    # Ensure input is a list
    if isinstance(histories, dict):
        histories = [histories]
        
    n_runs = len(histories)
    keys = ['loss', 'bc']
    stats = {}

    # Calculate mean and std for each metric
    for key in keys:
        # Stack values from all runs: shape (n_runs, n_iterations)
        data = np.stack([h[key] for h in histories], axis=0)
        stats[key] = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0)
        }
    
    iterations = np.arange(len(stats['loss']['mean']))
    
    plt.figure(figsize=(12, 4))
    
    # Plot L2 Loss
    plt.subplot(1, 2, 1)
    mean = stats['loss']['mean']
    std = stats['loss']['std']
    
    plt.semilogy(iterations, mean, label='Mean Loss')
    plt.fill_between(iterations, mean - std, mean + std, alpha=0.3, label='$\\pm$ 1 Std Dev')
    
    plt.xlabel('Uzawa Iterations')
    plt.title(f'$L^2$ Loss in $\\Omega$')
    plt.legend()
    plt.grid(True)
    
    # Plot Boundary Loss
    plt.subplot(1, 2, 2)
    mean_bc = stats['bc']['mean']
    std_bc = stats['bc']['std']
    
    plt.semilogy(iterations, mean_bc, label='Mean Boundary Loss', color='orange')
    plt.fill_between(iterations, mean_bc - std_bc, mean_bc + std_bc, color='orange', alpha=0.3, label='$\\pm$ 1 Std Dev')
    
    plt.xlabel('Uzawa Iterations')
    plt.title(f'$L^2$ Loss in $\\partial \\Omega$')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_history(history):
    """Plots the Dirichlet and Boundary energy history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(history['loss'], label='Loss')
    plt.xlabel('Uzawa Iterations')
    plt.title('$L^2$ Loss in $\\Omega$')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(history['bc'], label='Boundary Loss', color='orange')
    plt.xlabel('Uzawa Iterations')
    plt.title(f'$L^2$ Loss in $\\partial \\Omega$')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def train(epsilon, model, gamma, rho, n_sgd, n_uz, eta, beta, plot_interval=1, method='ritz'):
    """
    Args:
        epsilon (float): PDE parameter.
        gamma (float): Boundary parameter.
        rho (float): Uzawa parameter.
        n_sgd: SGD step count.
        n_uz: Uzawa step count.
        eta: Learning rate.
        beta: Mini-batch size.
        method: ritz or pinn
    """
    optimizer = optim.Adam(model.parameters(), lr=eta)

    # Initialize lambdas (scalars)
    lambda_0 = torch.tensor(0.0)
    lambda_1 = torch.tensor(0.0)

    x_batch = torch.linspace(0, 1, beta + 1)[1:-1].view(beta - 1, 1)
    x_batch.requires_grad_()

    history = {'loss': [], 'bc': [], 'tot': []}
    
    for k in range(n_uz):

        for m in range(n_sgd):
            
            # Compute Energies
            if method == 'ritz':
                energy = torch.mean(dirichlet(epsilon, model, x_batch))
            else: # method == 'pinn'
                energy = torch.mean(pinn(epsilon, model, x_batch))
            
            bc_loss = boundary(model)
            lag = lagrange(model, lambda_0, lambda_1)
            
            # Total Energy: Energy + gamma * Boundary - Lagrange
            energy_tot = energy + bc_loss - lag
            
            optimizer.zero_grad()
            energy_tot.backward()
            optimizer.step()

        loss = L2_distance(epsilon, model, x_batch)
        # Plot Solution if k divides plot_interval
        if plot_interval > 0 and k % plot_interval == 0:
            print(f"Iteration {k}: Loss={loss.item():.2f}; Boundary loss={bc_loss.item():.2f}")
            plot_solution(model, k, epsilon)
        
        # Update lambda (no gradient)
        with torch.no_grad():
            u_0, u_1 = evaluate(model, [0, 1])
            lambda_0 -= rho * u_0.item()
            lambda_1 -= rho * u_1.item()

        # Log average energy for this Uzawa step
        history["loss"].append(loss.item())
        history["bc"].append(bc_loss.item())
        history["tot"].append(energy_tot.item())
        
    print(f"Final Iteration: Loss={loss.item():.2f}; Boundary loss={bc_loss.item():.2f}")
    plot_solution(model, n_uz, epsilon)

    return model, history

if __name__ == "__main__":

    base = FFNN(width=40, depth=5)

    ffnn = copy.deepcopy(base)
    _, history = train(epsilon=1e-1, model=ffnn, gamma=2, rho=0.1, n_sgd=40, n_uz=100, eta=1e-3, beta=1000, plot_interval=0)
    plot_history(history)

    ffnn = copy.deepcopy(base)
    _, history = train(epsilon=1e-1, model=ffnn, gamma=0, rho=0.1, n_sgd=40, n_uz=100, eta=1e-3, beta=1000, plot_interval=0)
    plot_history(history)

    ffnn = copy.deepcopy(base)
    histories = []
    for i in range (10):
        _, history = train(epsilon=1e-3, model=ffnn, gamma=0, rho=0.1, n_sgd=40, n_uz=100, eta=1e-3, beta=1000, plot_interval=0)
        histories.append(history)
    plot_history(history)
    plot_histories(histories)

    ffnn = copy.deepcopy(base)
    histories = []
    for i in range (10):
        _, history = train(epsilon=1e-3, model=ffnn, gamma=2, rho=0.1, n_sgd=40, n_uz=500, eta=1e-3, beta=1000, plot_interval=0)
        histories.append(history)
    plot_history(history)
    plot_histories(histories)

    ffnn = copy.deepcopy(base)
    _, history = train(epsilon=1e-3, model=ffnn, gamma=0, rho=0.01, n_sgd=40, n_uz=100, eta=1e-3, beta=1000, plot_interval=0)
    plot_history(history)

    ffnn = copy.deepcopy(base)
    _, history = train(epsilon=1e-3, model=ffnn, gamma=2, rho=0.01, n_sgd=40, n_uz=100, eta=1e-3, beta=1000, plot_interval=0)
    plot_history(history)

    ffnn = copy.deepcopy(base)
    _, history = train(epsilon=1e-3, model=ffnn, gamma=2, rho=0.1, n_sgd=40, n_uz=100, eta=1e-3, beta=1000, plot_interval=0, method="pinn")
    plot_history(history)
