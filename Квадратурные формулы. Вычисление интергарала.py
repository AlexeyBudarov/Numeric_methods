import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import warnings
warnings.filterwarnings("ignore")

# def function(x):
#     return 1.3*np.cos(3.5*x) * np.exp(2*x/3) + 6*np.sin(4.5*x) * np.exp(-x/8) + 5*x
def function(t):
    b = 3.2
    return 1.3 * np.cos(3.5 * (b - t)) * np.exp(2 * (b - t) / 3) + 6 * np.sin(4.5 * (b - t)) * np.exp(-(b - t) / 8) + 5 * (b - t)


def Integ_function(t):
    b = 3.2
    return (1.3*np.cos(3.5*(b-t))*np.exp(2*(b-t)/3) + 6*np.sin(4.5*(b-t))*np.exp(-(b-t)/8) + 5*(b-t))/t**0.25

# def function(t):
#     return 3 * np.cos(1.6 - 0.5 * t) * np.exp((3.2 - t)/4) + 5 * np.sin(8 - 2.5 * t) * np.exp((t - 3.2)/3) + 6.4 - 2 * t
#
# def Integ_function(t):
#     f = 3 * np.cos(1.6 - 0.5 * t) * np.exp((3.2 - t)/4) + 5 * np.sin(8 - 2.5 * t) * np.exp((t - 3.2)/3) + 6.4 - 2 * t
#     return f / t**0.25


def get_func_moments(a, b, k):
    return 4/(4 * k + 3) * (b**(k + 3/4) - a**(k + 3/4))
    # return 4/(4*k + 3) * (b-a)**(k + 3/4)
def ISquareProcess(cnt_nodes, start_params, flag = False):

    a, b, betta = start_params
    n = cnt_nodes
    x_nodes = np.linspace(a, b, cnt_nodes)

    func_moments = [get_func_moments(a, b, i) for i in range(n)]
    # A_matrix = np.array([[x_nodes[j]**i for j in range(n)]for i in range(n)])
    A_matrix = np.vander(x_nodes, increasing=True).T

    A_array = np.linalg.solve(A_matrix, func_moments)
    if flag == True:
        return A_array
    return sum(A_array*np.array([function(t) for t in x_nodes])), x_nodes

def GaussSquareProcess(cnt_nodes, start_params, flag = False):
    n = cnt_nodes
    a, b, betta = start_params
    all_func_moments = np.array([get_func_moments(a, b, i) for i in range(2*n)])

    a_find_vector_b = -all_func_moments[n:]
    a_find_matrix_A = np.array([[all_func_moments[j+i] for j in range(n)]for i in range(n)])

    a_array = np.linalg.solve(a_find_matrix_A, a_find_vector_b)
    a_array = np.flip(a_array, axis=0)
    a_array = np.insert(a_array, 0,1)
    roots_nodes = np.roots(a_array)
    roots_nodes.sort()
    # print(roots_nodes)

    A_find_matrix_A = np.array([[roots_nodes[j]**i for j in range(n)] for i in range(n)])
    # A_find_matrix_A = np.vander(roots_nodes, increasing=True).T

    A_find_vector_b = np.array(all_func_moments[:n])

    A_array = np.linalg.solve(A_find_matrix_A, A_find_vector_b)
    if flag == True:
        return A_array
    # print(A_array)
    # if roots_nodes[0] < a or roots_nodes[-1] > b:
    #     print(f'Error for {n} nodes, {A_array}')
    return sum(A_array*np.array([function(t) for t in roots_nodes])), roots_nodes

def Composite_Quadrature_forms(start_params, k, cnt_nodes, flag = 'Newton'):
    a, b, betta = start_params
    sections = np.linspace(a, b, k)
    result = 0
    all_sections = [] #Может понадобиться для постороения графиков
    if flag == 'Newton':
        for i in range(len(sections) - 1):
            a, b = sections[i], sections[i+1]

            N_result_on_section, N_x_nodes_on_section = ISquareProcess(cnt_nodes, (a, b, betta))
            result += N_result_on_section
            all_sections.append(N_x_nodes_on_section)
        return result, all_sections
    else:
        for i in range(len(sections) - 1):
            a, b = sections[i], sections[i + 1]
            G_result_on_section, G_x_nodes_on_section = GaussSquareProcess(cnt_nodes, (a, b, betta))
            result += G_result_on_section
            all_sections.append(G_x_nodes_on_section)
        return result, all_sections





def compare_error_rate(my_result, start_params, flag = 'Newton_Cotes'):
    a, b, betta = start_params
    scipy_result = integrate.quad(Integ_function, a, b, epsabs=1.0e-14, epsrel=1.0e-13)
    print(f"{flag} error rate: {abs(my_result - scipy_result[0])}")
    return 0

# |An| от количества узлов
def get_graph_1(cnt_nodes, start_params):
    arr_nodes = np.arange(1, cnt_nodes+1)
    arr_nodes_2 = np.arange(1, 10)
    Newton_coeff = np.log10([sum(abs(ISquareProcess(cnt, start_params, True))) for cnt in range(1, cnt_nodes+1)])
    Gauss_coeff = np.log10([sum(abs(GaussSquareProcess(cnt, start_params, True))) for cnt in range(1, 10)])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(arr_nodes, Newton_coeff, color = 'green', linewidth = 2, label = 'Newton_sum_of_abs_coeff')
    ax.plot(arr_nodes_2, Gauss_coeff, color = 'red', linewidth = 1, label = 'Gauss_sum_of_abs_coeff')
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Sum of |Am|')
    ax.legend()
    ax.grid('--')
    plt.show()

def get_graph_1_new(cnt_nodes, start_params):
    arr_nodes = np.arange(1, cnt_nodes + 1)
    Gauss_coeff = np.log10([sum(abs(GaussSquareProcess(cnt, start_params, True))) for cnt in range(1, cnt_nodes + 1)])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(arr_nodes, Gauss_coeff, color='red', linewidth=1, label='Gauss_sum_of_abs_coeff')
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Sum of |Am|')
    ax.legend()
    ax.grid('--')
    plt.show()


# Абсолютная погрешность от количества узлов
def get_graph_2(cnt_nodes, start_params):
    arr_nodes = np.arange(1, cnt_nodes+1)
    arr_nodes_2 = np.arange(1, 10)
    scipy_result = integrate.quad(Integ_function, a, b, epsabs=1.0e-14, epsrel=1.0e-13)

    Newton_abs_error = np.log10([abs(ISquareProcess(cnt, start_params)[0] - scipy_result[0]) for cnt in range(1, cnt_nodes+1)])
    Gauss_abs_error = np.log10([abs(GaussSquareProcess(cnt, start_params)[0] - scipy_result[0]) for cnt in range(1, 10)])

    fig, axs = plt.subplots(figsize = (10, 5))
    axs.plot(arr_nodes, Newton_abs_error, color = 'green', linewidth = 3, label = 'Newton_abs_error')
    axs.plot(arr_nodes_2, Gauss_abs_error, color = 'red', linewidth = 2, label = 'Gauss_abs_error')
    axs.set_xlabel('Number of nodes')
    axs.set_ylabel('Absolute error rate')
    axs.legend()
    axs.grid('--')
    plt.show()

# Абсолютная погрешность от количества разбиений
def get_graph_3(start_params, params_to_compose = (15, 3)):
    a, b, betta = start_params
    k, cnt_compose_nodes = params_to_compose
    arr_k = np.arange(1, k + 1)
    scipy_result = integrate.quad(Integ_function, a, b, epsabs=1.0e-14, epsrel=1.0e-13)

    Compose_Gauss = np.log10([abs(Composite_Quadrature_forms(start_params, k_cnt, cnt_compose_nodes, 'Gauss')[0] - scipy_result[0]) for k_cnt in arr_k])
    Compose_Newton = np.log10([abs(Composite_Quadrature_forms(start_params, k_cnt, cnt_compose_nodes, 'Newton')[0] - scipy_result[0]) for k_cnt in arr_k])

    fig, axs = plt.subplots(figsize=(10, 5))
    axs.plot(arr_k, Compose_Newton, 'pr-', label='Newton_composite_abs_error')
    axs.plot(arr_k, Compose_Gauss, 'pg-', label='Gauss_composite_abs_error')
    axs.set_xlabel('Number of division')
    axs.set_ylabel('Absolute error rate')
    axs.legend()
    axs.grid('--')
    plt.show()

def get_opt_h(start_params, params_to_compose, flag = 'Newton'):
    a, b, betta = start_params
    k, cnt_nodes, eps = params_to_compose
    L = 1.5
    h = (b - a) / k
    def get_m(h): # Процесс Эйткена
        S_h1 = Composite_Quadrature_forms(start_params, int(np.ceil((b - a) / h)), cnt_nodes, flag)[0]
        S_h2 = Composite_Quadrature_forms(start_params, int(np.ceil(L * (b - a) / h)), cnt_nodes, flag)[0]
        S_h3 = Composite_Quadrature_forms(start_params, int(np.ceil(L ** 2 * (b - a) / h)), cnt_nodes, flag)[0]
        m = - np.log(abs(S_h3 - S_h2) / abs(S_h2 - S_h1)) / np.log(L)
        return m
    R_h = 1
    i = 0
    while abs(R_h) > eps:
        i += 1
        m = get_m(h) # Процесс Эйткена
        S_h1 = Composite_Quadrature_forms(start_params, int(np.ceil((b - a) / h)), cnt_nodes, flag)[0]
        S_h2 = Composite_Quadrature_forms(start_params, int(np.ceil(L * (b - a) / h)), cnt_nodes, flag)[0]
        Cm = (abs(S_h2 - S_h1))/(h**m * (1 - L**(-m)))
        R_h = (abs(S_h2 - S_h1))/(1 - L**(-m))     # Правило Рунге
        print(f"i = {i} h = {'%.8f' % h}, k = {int(np.ceil((b - a) / h))} m = {'%.8f' % m} Cm = {'%.8f' % Cm} R_h = {R_h}")
        h = h / L
    # h *= 0.95
    print()
    new_k = int(np.ceil((b - a) / h))
    return new_k




# All parameters
a = 0.7
b = 3.2
alpha = 0
betta = 0.25


start_params = (a, b, betta)
cnt_nodes = 20

# Newton-Cotes ИКФ
# result_Newton, x_nodes_Newton = ISquareProcess(cnt_nodes, start_params)
# compare_error_rate(result_Newton, start_params)
#
# Gauss ИКФ
result_Gauss, new_nodes_Gauss = GaussSquareProcess(cnt_nodes, start_params)
# compare_error_rate(result_Gauss, start_params, 'Gauss')

# get_graph_1(cnt_nodes, start_params) # |An| от количества узлов
# get_graph_2(cnt_nodes, start_params) # Абсолютная погрешность от количества узлов
get_graph_1_new(100, start_params)


Compose_cnt_nodes = 3
eps = 10**-8
print('Compose Gauss process')
k_gauss = get_opt_h(start_params, (1, Compose_cnt_nodes, eps),  'Gauss')
print('Compose Newton-Cotes process')
k_newton = get_opt_h(start_params, (1, Compose_cnt_nodes, eps), 'Newton')
# # print(k_gauss)
# print(f"h_gauss = {(b - a)/ k_gauss}")
# # print(k_newton)
# print(f"h_newton = {(b - a)/ k_newton}")


result_GaussCompose, x_nodes_GaussCompose = Composite_Quadrature_forms(start_params, k_gauss, Compose_cnt_nodes, 'Gauss')

result_NewtonCompose, x_nodes_NewtonCompose = Composite_Quadrature_forms(start_params, k_newton, Compose_cnt_nodes)


# compare_error_rate(result_GaussCompose, start_params, 'Compose Gauss process')


# compare_error_rate(result_NewtonCompose, start_params, 'Compose Newton-Cotes process')
# print(x_nodes_NewtonCompose)

# get_graph_3(start_params, (100, 3)) # От количества разбиений