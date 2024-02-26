from sympy import symbols, diff
import random
import numpy as np

# gradient function
def simple_gradient(loss_fn, x_min, x_max, variables: list, max_iter: int, lr: float, min_step_size: float):
    val_per_iter = {}

    x_result = {}
    for x in variables:
        x_result[x] = round(random.uniform(x_min, x_max), 2)

    x_result = {variables[0]: 0, variables[1]: 8}  # comment if ypu want a random starting point
    trajectory = np.array([list(x_result.values())])

    gradient = []
    for x in variables:
        gradient.append(diff(loss_fn, x))

    print(gradient)
    x_step_size = [1 for i in range(len(variables))]
    it = 1
    while(it < max_iter):
        val_per_iter[it] = loss_fn.subs(x_result)
        trajectory = np.append(trajectory, [np.array(list(x_result.values()))], axis=0)
        if max([abs(ss) for ss in x_step_size]) < min_step_size: break
        for i in range(len(variables)):
            x_step_size[i] = gradient[i].subs(x_result) * lr
            x_result[variables[i]] -= x_step_size[i]

        it += 1

    return x_result, trajectory, val_per_iter
