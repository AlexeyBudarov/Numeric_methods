from Newton_s_and_Lagrange_s_polynomials import get_interpolation_grid, get_graphics, Max_Deviation
import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return x**3 - np.exp(x) + 1

# def function(x):
#     return np.sin(np.log(x))

# def function(x):
#     return np.sqrt(x) + np.cos(x)

# interval = [0, 5]

# interval = [-5, 10]
interval = [-1, 4]
# interval = [-1,1]

def Linear_splain(x, n_nodes,  interval, function, flag = 'not_opt'):  #
    grid = get_interpolation_grid(function, interval, n_nodes, flag)
    n_nodes = len(grid[0])
    coeff_array = np.array([])
    for i in range(n_nodes - 1):
        A = [[grid[0][i], 1], [grid[0][i+1], 1]]
        b = [grid[1][i], grid[1][i+1]]
        coeff_array = np.append(coeff_array, np.linalg.solve(A, b))
    coeff_array = coeff_array.reshape(-1, 2)

    for k in range(n_nodes-1):
        if (x >= grid[0][k]) & (x <= grid[0][k+1]):
            return coeff_array[k][0]*x + coeff_array[k][1]


def Quadr_splain(x, n_nodes,  interval, function, flag = 'not_opt'):
    grid = get_interpolation_grid(function, interval, n_nodes, flag)
    matrix = []
    y = []
    n_nodes = len(grid[0])
    for i in range(n_nodes - 1):
        str_1 = str_2 = str_3 = [0]*3*i

        str_1 = np.append(str_1, [grid[0][i]**2, grid[0][i], 1])
        str_1 = np.append(str_1, [0] * (3*(n_nodes - 1) - len(str_1)))

        str_2 = np.append(str_2, [grid[0][i+1] ** 2, grid[0][i+1], 1])
        str_2 = np.append(str_2, [0] * (3*(n_nodes - 1) - len(str_2)))

        if i == (n_nodes-2):
            str_3 = np.append(str_3, [2 * grid[0][i + 1], 1, 0])
        else:
            str_3 = np.append(str_3, [2*grid[0][i + 1], 1, 0, -2 * grid[0][i + 1], -1, 0])
            str_3 = np.append(str_3, [0]*(3*(n_nodes - 1) - len(str_3)))

        y = np.append(y, [grid[1][i], grid[1][i+1], 0])
        matrix = np.append(matrix, [str_1, str_2, str_3])
    matrix = matrix.reshape(-1, 3*(n_nodes - 1))
    coeff_array = np.linalg.solve(matrix, y).reshape(-1, 3)

    for k in range(n_nodes-1):
        if (x >= grid[0][k]) & (x <= grid[0][k+1]):
            return coeff_array[k][0]*x**2 + coeff_array[k][1]*x + coeff_array[k][2]



def Cub_splain(x, n_nodes,  interval, function, flag = 'not_opt'):
    grid = get_interpolation_grid(function, interval, n_nodes, flag)
    n_nodes = len(grid[0])
    h_array = [grid[0][i+1] - grid[0][i] for i in range(n_nodes-1)]
    H_array = np.zeros((n_nodes-2, n_nodes-2))
    for i in range(n_nodes - 3):

        H_array[i][i] = 2*(h_array[i] + h_array[i+1])
        H_array[i][i+1] = h_array[i+1]
        H_array[i+1][i] = h_array[i+1]
        if i == n_nodes - 4:
            H_array[n_nodes - 3][n_nodes - 3] = 2 * (h_array[i + 1] + h_array[i + 2])

    gamma = [6*((grid[1][i+1] - grid[1][i])/h_array[i]) for i in range(1, n_nodes-1)]
    y_second = np.linalg.solve(H_array, gamma)
    y_second = np.append(y_second, 0)
    y_second = np.insert(y_second, 0, 0)

    y_first = [(grid[1][i+1] - grid[1][i])/h_array[i] - y_second[i+1]*h_array[i]/6 - y_second[i]*h_array[i]/3
               for i in range(n_nodes - 1)]

    for k in range(n_nodes-1):
        if (x >= grid[0][k]) & (x <= grid[0][k+1]):
            return grid[1][k] + \
                   y_first[k]*(x-grid[0][k]) + \
                   y_second[k]*((x-grid[0][k])**2)/2 + \
                   (y_second[k+1] - y_second[k])*((x-grid[0][k])**3) / (6*h_array[k])





get_graphics(function, interval, Linear_splain)
get_graphics(function, interval, Quadr_splain)
get_graphics(function, interval, Cub_splain)

# opt_result = Max_Deviation(function, interval, 'opt', Linear_splain)
# not_opt_result = Max_Deviation(function, interval, 'not_opt', Linear_splain)
# print(f"Optimal result: {opt_result} \nNot optimal result: {not_opt_result}")


# opt_result = Max_Deviation(function, interval, 'opt', Quadr_splain)
# not_opt_result = Max_Deviation(function, interval, 'not_opt', Quadr_splain)
# print(f"Optimal result: {opt_result} \nNot optimal result: {not_opt_result}")


# opt_result = Max_Deviation(function, interval, 'opt', Cub_splain)
# not_opt_result = Max_Deviation(function, interval, 'not_opt', Cub_splain)
# print(f"Optimal result: {opt_result} \nNot optimal result: {not_opt_result}")