import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


def function(x):
    return x**3 - np.exp(x) + 1

# def function(x):
#     return np.sqrt(x) + np.cos(x)

# def function(x):
#     return np.sin(np.log(x))

interval = [-5, 10]
# interval = [1, 10]
# interval = [0,5]

def get_interpolation_grid(function, interval, n_nodes, flag = 'not_opt'):
    if flag == 'not_opt':
        x_values = np.linspace(interval[0], interval[1], n_nodes)
        result = [[x for x in x_values], [function(x) for x in x_values]]
        return result

    elif flag == 'opt':
        x_values = [0.5*((interval[1] - interval[0])*
                        np.cos((2*i + 1)/(2*(n_nodes + 1))*np.pi) +
                        (interval[1] + interval[0])) for i in range(n_nodes)]
        if interval[0] < x_values[n_nodes-1]:
            x_values = np.append(x_values, interval[0])
        if interval[1] > x_values[0]:
            x_values = np.append(x_values, interval[1])
            x_values.sort()
        result = [[x for x in x_values], [function(x) for x in x_values]]
        return result

def Lagrange_polinomial(x, n_nodes,  interval, function, flag = 'not_opt'):
    inter_grid = get_interpolation_grid(function, interval, n_nodes, flag)
    n_nodes = len(inter_grid[0])
    P_x = 0
    for j in range(n_nodes):
        lgr_grid = [inter_grid[0][k] for k in range(n_nodes) if k != j]
        lgr_mult = 1
        for i in range(len(lgr_grid)):
            lgr_mult *= (x - lgr_grid[i])/(inter_grid[0][j] - lgr_grid[i])
        P_x += lgr_mult*inter_grid[1][j]
    return P_x


def Newton_polinomial(x, n_nodes, interval, function, flag = 'not_opt'):
    grid = get_interpolation_grid(function, interval, n_nodes, flag)
    N_p = grid[1][0]
    g_grid = grid[0]
    n_nodes = len(grid[0])
    temp_nodes = 1
    for k in range(1, n_nodes):
        f_g = 0
        temp = g_grid[:k + 1]
        temp_nodes *= (x - temp[k - 1])
        for i in range(len(temp)):
            g_temp = 1
            for j in range(len(temp)):
                if i != j:
                    g_temp *= 1 / (temp[i] - temp[j])
            f_g += g_temp * grid[1][i]
        N_p += f_g * temp_nodes
    return N_p


def Max_Deviation(function, interval, flag, Polinomial):
    t_range = np.arange(interval[0], interval[1], 0.1)
    # m = len(t_range)
    if (Polinomial == Lagrange_polinomial) or (Polinomial == Newton_polinomial):
        n = [3,10,30,50]
    else:
        n = [10,30,50,100]

    max_deviation = []
    y_function = [function(t) for t in t_range]
    for i in range(len(n)):
        y_PL = [Polinomial(t, n[i], interval, function, flag) for t in t_range]
        result = [abs(y_function[i] - y_PL[i]) for i in range(len(y_PL))]
        max_deviation = np.append(max_deviation, max(result))
    return max_deviation



def get_graphics(function, interval, Polinomial):
    x_range = np.arange(interval[0], interval[1], 0.1)
    y_function = [function(x) for x in x_range]
    flag = 'not_opt'
    if (Polinomial == Lagrange_polinomial) or (Polinomial == Newton_polinomial):
        n = [3,10,30,50]
    else:
        n = [10,30,50,100]

    y_PL3 =  [Polinomial(x, n[0],  interval, function, flag) for x in x_range]
    y_PL10 = [Polinomial(x, n[1], interval, function, flag) for x in x_range]
    y_PL30 = [Polinomial(x, n[2], interval, function, flag) for x in x_range]
    y_PL50 = [Polinomial(x, n[3], interval, function, flag) for x in x_range]
    plt.subplot(211)
    plt.title('Not_optimized')
    if (Polinomial == Lagrange_polinomial) or (Polinomial == Newton_polinomial):
        plt.axis([-6, 8, -200, 150])
        print()

    plt.plot(x_range, y_function, 'c--*',
             x_range,y_PL3, 'g--',
             x_range, y_PL10, 'r-',
             x_range, y_PL30, 'y-',
             x_range, y_PL50, 'k--',)

    flag = 'opt'
    y_PL3_opt = [Polinomial(x, n[0], interval, function, flag) for x in x_range]
    y_PL10_opt = [Polinomial(x, n[1], interval, function, flag) for x in x_range]
    y_PL30_opt = [Polinomial(x, n[2], interval, function, flag) for x in x_range]
    y_PL50_opt = [Polinomial(x, n[3], interval, function, flag) for x in x_range]
    plt.subplot(212)
    plt.title('Optimazed')
    if (Polinomial == Lagrange_polinomial) or (Polinomial == Newton_polinomial):
        print()
        plt.axis([-6, 8, -200, 150])
    plt.plot(x_range, y_function, 'c--^',
             x_range, y_PL3_opt, 'g--',
             x_range, y_PL10_opt, 'r-',
             x_range, y_PL30_opt, 'y--',
             x_range, y_PL50_opt, 'k--', )
    plt.show()
#
# get_graphics(function,interval, Newton_polinomial)
# get_graphics(function, interval, Lagrange_polinomial)

opt_result = Max_Deviation(function,interval,  'opt', Lagrange_polinomial)
not_opt_result = Max_Deviation(function, interval,  'not_opt', Lagrange_polinomial)
print(f"Optimal result: {opt_result} \nNot optimal result: {not_opt_result}")

# opt_result = Max_Deviation(function, interval,  'opt', Newton_polinomial)
# not_opt_result = Max_Deviation(function, interval, 'not_opt', Newton_polinomial)
# print(f"Optimal result: {opt_result} \n Not optimal result: {not_opt_result}")








