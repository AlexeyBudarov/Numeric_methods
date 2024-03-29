import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 5)
#
c = 0.55
A = 2
B = -1
C = 2
#
# c = 0.45
# A = -2
# B = 2
# C = -2
section = np.array([0, 5])

y_start = np.array([1, 1, A, 1])
x_start = section[0]
eps = 10**-5
p = len(y_start)


answer = lambda x: np.array([np.exp(np.sin(x**2)), np.exp(B*np.sin(x**2)), C*np.sin(x**2) + A, np.cos(x**2)])

def f(x, y):
    y1, y2, y3, y4 = y
    return np.array([2*x*np.power(y2, (1/B))*y4,
                     2*B*x*np.exp(B/C * (y3 - A))*y4,
                     2*C*x*y4,
                     -2*x*np.log(y1)])




# A = 1/35
# B = 1/10
# c = 1/17
# eps = 10**-4
# section = np.array([0, np.pi])
# y_start = np.array([B*np.pi, A*np.pi])
# def f(x, y):
#     y = np.array(y)
#     return np.array([A * y[1], -B * y[0]])





def get_start_h(section, p, eps, params):
    eps = 10**-7
    x_start, y_start = params
    delta = (1/(max(x_start, section[1])))**(p+1) + np.linalg.norm(f(x_start, y_start))**(p + 1)
    h = (eps/delta)**(1/(p+1))
    x_start_2 = x_start + h
    y_start_2 = y_start + h*f(x_start_2, y_start)

    delta = (1/(max(x_start_2, section[1])))**(p+1) + np.linalg.norm(f(x_start_2, y_start_2))**(p + 1)
    h_new = (eps/delta)**(1/(p+1))

    if h <= h_new:
        return x_start, y_start, h
    else:
        return x_start_2, y_start_2, h_new



def opponent_method(eps, params, section = section, f = f, flag = False):

    x_n, y_opp, h = params

    def temp_func(x_n, h:float,y_opp:np.ndarray, f):
        k1_opp = f(x_n, y_opp)
        k2_opp = f(x_n + 1/2*h, y_opp + 1/2*h*k1_opp)
        k3_opp = f(x_n + h, y_opp - h*k1_opp + 2*h*k2_opp)

        return y_opp + h*(1/6*k1_opp + 4/6*k2_opp + 1/6*k3_opp)



    x_for_graph = np.array([])
    error_for_graph = np.array([])

    if flag == False:
        while x_n < section[1]:
            y_opp = temp_func(x_n, h, y_opp, f)
            x_n += h
        # print(f'my:{y_opp} - ans:{answer(section[1])}')
        return y_opp
    else:
        while x_n < section[1]:
            x_for_graph = np.append(x_for_graph, x_n)
            error_for_graph = np.append(error_for_graph, np.log10(np.linalg.norm(answer(x_n) - y_opp)))
            y_opp = temp_func(x_n, h, y_opp, f)
            x_n += h
        return x_for_graph, error_for_graph





def two_stages_method(eps, params, section = section, f = f, flag = False):

    def temp_func(x_n:float, h:float,y_n:np.ndarray, f, c:float = c):

        b1 = 1 - 1 / (2 * c)
        b2 = 1 / (2 * c)
        a = c

        K1 = f(x_n, y_n)
        K2 = f(x_n + c * h, y_n + h * a * K1)
        return y_n + h * (b1 * K1 + b2 * K2)

    x_n, y_n, h = params
    x_for_graph = np.array([])
    error_for_graph = np.array([])
    if flag == False:
        while x_n < section[1]:
            y_n = temp_func(x_n, h, y_n, f, c)
            x_n += h
        return y_n
    else:
        while x_n < section[1]:
            x_for_graph = np.append(x_for_graph, x_n)
            error_for_graph = np.append(error_for_graph, np.log10(np.linalg.norm(answer(x_n) - y_n)))
            y_n = temp_func(x_n, h, y_n, f, c)
            x_n += h
        return x_for_graph, error_for_graph

    # print(f'my:{y_n} - ans:{answer(section[1])}')




def find_optimal_h(eps, x_start, y_start, flag = 'two_stages_method'):
    R_i = 1
    x_start, y_start, h = get_start_h(section, p, eps, (x_start, y_start))
    # h = 1/2**10
    if flag == 'two_stages_method':
        while np.linalg.norm(R_i) >= eps:
            h = h * (eps / np.linalg.norm(R_i)) ** (1 / p)
            y_n = two_stages_method(eps, (x_start, y_start, h))
            y_2n = two_stages_method(eps, (x_start, y_start, h/2))
            R_i = (y_2n - y_n) / (1 - 2 ** -p)

            print(h)
        return h
    else:
        while np.linalg.norm(R_i) >= eps:
            h = h * (eps / np.linalg.norm(R_i)) ** (1 / p)
            y_n = opponent_method(eps, (x_start, y_start, h))
            y_2n = opponent_method(eps, (x_start, y_start, h/2))
            R_i = (y_2n - y_n) / (1 - 2 ** -p)
            print(h)
        return h
def get_graph_1():

    h_array = np.array([1/2**k for k in range(6, 12)])


    error_two_stages_array = np.log10([np.linalg.norm(two_stages_method(eps, (x_start, y_start, h)) - answer(section[1])) for h in h_array])

    error_opponent_array = np.log10([np.linalg.norm(opponent_method(eps, (x_start, y_start, h)) - answer(section[1])) for h in h_array])
    fig, ax = plt.subplots()
    ax.plot(np.log10(h_array), error_two_stages_array, 'r', label= 'error_two_stages_method')
    ax.plot(np.log10(h_array), error_opponent_array, 'g', label= 'error_opponent_method')
    ax.plot(np.log10(h_array), 2*np.log10(h_array) + 2.4, 'r--', label = '2x')
    ax.plot(np.log10(h_array), 3 * np.log10(h_array)+ 2.4, 'g--', label = '3x')
    ax.legend()
    ax.set_xlabel('h_values')
    ax.set_ylabel('method_errors')
    plt.show()
get_graph_1()


def get_graph_2(h_opt_two_stages, h_opt_opponent):
    x1, y1 = two_stages_method(eps, (x_start, y_start, h_opt_two_stages), section, f, True)
    x2, y2 = opponent_method(eps, (x_start, y_start, h_opt_opponent), section, f, True)

    fig, ax = plt.subplots()
    ax.plot(x1, y1, label = "two_stages_method")
    ax.plot(x2, y2, label = 'opponent_method')
    ax.legend()
    ax.set_xlabel('x_values')
    ax.set_ylabel('error_optimal_step')
    plt.show()

h_opt_two_stages = find_optimal_h(eps, x_start, y_start, 'two_stages_method')
#
h_opt_opponent = find_optimal_h(eps, x_start, y_start, 'opponent_method')
print(h_opt_opponent)
print(h_opt_two_stages)
get_graph_2(h_opt_two_stages, h_opt_opponent)

    # params = get_start_h(section, p, eps, (x_start, y_start))
    # params_2 = (x_start, y_start, 0.0001)
    #
    # print(h_opt_two_stages)
    # print(h_opt_opponent)
    #
    # res_1 = two_stages_method(eps, (x_start, y_start, h_opt_two_stages))
    # res_2 = opponent_method(eps, (x_start, y_start, 1/2**10))
    # print(res_1)
    # print(res_2)
    # print(np.linalg.norm(res_1 - answer(section[1])))
    # print(np.linalg.norm(res_2 - answer(section[1])))










def two_stages_temp(x_n: float, h: float, y_n: np.ndarray, f, c: float = c):

    b1 = 1 - 1 / (2 * c)
    b2 = 1 / (2 * c)
    a = c
    K1 = f(x_n+h, y_n)
    K2 = f(x_n + c * h, y_n + h * a * K1)
    return y_n + h * (b1 * K1 + b2 * K2)

def opponent_temp(x_n: float, h:float,y_opp:np.ndarray, f):
    k1_opp = f(x_n, y_opp)
    k2_opp = f(x_n + 1 / 2 * h, y_opp + 1 / 2 * h * k1_opp)
    k3_opp = f(x_n + h, y_opp - h * k1_opp + 2 * h * k2_opp)

    return y_opp + h * (1 / 6 * k1_opp + 4 / 6 * k2_opp + 1 / 6 * k3_opp)




def auto_step_method(eps, x_start, y_start, section = section, f = f, method = two_stages_temp): # y1 - > h = x_n - x_(n-1) y2 -> h/2
    p = 2
    x_n, y_n, h = get_start_h(section, p, eps, (x_start, y_start))
    atol = 10 ** -12
    h_max = 0.5
    # x_n = x_start
    # y_n = y_start
    #
    # h = 10**-10
    for_graph = [[],[]]
    while x_n < section[1]:
        for_graph[0] += [x_n]
        for_graph[1] += [np.log10(np.linalg.norm(y_n - answer(section[1])))]

        y_1 = method(x_n, h, y_n, f)
        y_2 = method(x_n, h / 2, y_n, f)
        y_2 = method(x_n + h/2, h / 2, y_2, f)

        r_n = np.linalg.norm((y_2 - y_1) / (1 - 2 ** (-1 * p)))
        tol = eps * np.linalg.norm(y_n) + atol

        if r_n > tol * 2 ** p:
            h /= 2
            if x_n + h > section[1]:
                break
            continue

        elif tol < r_n <= tol * 2 ** p:
            h /= 2
            y_n = y_2

        elif tol / (2 ** (p + 1)) <= r_n <= tol:
            y_n = y_1

        else:
            h = min(2 * h, h_max)
            y_n = y_1

        x_n += h
        # print(h)
    return y_n, for_graph



def get_graph_3():
    rtol = 10 ** -6
    temp_1 = auto_step_method(rtol, x_start, y_start, section, f, two_stages_temp)[1]
    temp_2 = auto_step_method(rtol, x_start, y_start, section, f, opponent_temp)[1]
    fig, ax = plt.subplots()


    ax.plot(temp_1[0], temp_1[1], linewidth=2.5)
    ax.plot(temp_2[0], temp_2[1], linewidth=2.5)
    ax.legend(['Runge-Kutta method', 'Opponent_method'])
    ax.set_title('Зависимость нормы точной полной погрешности от независимой переменной')
    ax.set_xlabel('X')
    ax.set_ylabel(r'$\log10{|| y(x_{n}) - \overline{y_{n}} ||}$')

    plt.show()




get_graph_3()


























