from sympy import symbols, diff
import random
import numpy as np
from grad import simple_gradient
from helper_functions import visualize_fun, plot_loss_fn_value_per_iter

# define a loss function
x1, x2 = symbols('x1 x2')
f_1 = x1**2 + x2**2
f_matyas = 0.26*(x1**2 + x2**2) - 0.48*x1*x2
f_3 = x1**2 + x2**2 - x1*x2

# define the range
x_min = -10
x_max = 10



# -----------------------------
# first function
result, trajectory1, vpi1 = simple_gradient(f_1, -10, 10, [x1, x2], max_iter=10000, lr=0.001, min_step_size=0.0001)
print(result)

def f_1_vis(x1, x2):
    return x1**2 + x2**2

visualize_fun(f_1_vis, trajectory1)
plot_loss_fn_value_per_iter(vpi1, 0.01, 0.0001)




# second function
result, trajectory2, vpi2 = simple_gradient(f_matyas, -10, 10, [x1, x2], max_iter=10000, lr=0.001, min_step_size=0.0001)
print(result)


def f_matyas_vis(x1, x2):
    return 0.26*(x1**2 + x2**2) - 0.48*x1*x2

visualize_fun(f_matyas_vis, trajectory2)



# third function
result, trajectory3, vpi3 = simple_gradient(f_3, -10, 10, [x1, x2], max_iter=10000, lr=0.001, min_step_size=0.0001)
print(result)


def f_3(x1, x2):
    return x1**2 + x2**2 - x1*x2

visualize_fun(f_3, trajectory3)