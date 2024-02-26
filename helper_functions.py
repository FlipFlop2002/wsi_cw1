import matplotlib.pyplot as plt
import numpy as np


def visualize_fun(obj_fun: callable, trajectory: np.ndarray):
    min_x, min_y = trajectory[-1]
    start_x, start_y = trajectory[0]
    MIN_X = 10
    MAX_X = 10
    PLOT_STEP = 100

    x1 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    x2 = np.linspace(-MIN_X, MAX_X, PLOT_STEP)
    X1, X2 = np.meshgrid(x1, x2)
    Z = obj_fun(X1, X2)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X1, X2, Z, cmap='viridis', shading='auto')
    plt.colorbar(label='Objective Function Value')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Objective Function Visualization')

    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='Gradient Descent Steps', alpha=0.5)
    plt.plot(min_x, min_y, marker="o", color='yellow', label='Minimum found by gradient descent alg.')
    plt.plot(start_x, start_y, marker="o", color='blue', label='Starting point.')

    plt.legend()
    plt.show()


def plot_loss_fn_value_per_iter(values_iterations: dict, lr: float, min_step_size: float):
    plt.figure(figsize=(8, 6))
    plt.xlabel('iteration')
    plt.ylabel('loss function value')
    plt.title(f'Loss function value per iteration, lr: {lr}, min_step: {min_step_size}')

    plt.plot(list(values_iterations.keys()), list(values_iterations.values()))
    plt.show()
