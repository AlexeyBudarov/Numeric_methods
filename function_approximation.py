import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from Functions import matrix_mult, mtrx_T, zero_mtrx

# def function(x):
#     return x**3 + np.exp(x)

def function(x):
    return x**2 *np.cos(x)

# def function(x):
#     return x - np.sin(x)

interval = [-1, 1]
n = [1,2,3,4,5]


def difference(array):
    for j in range(len(array)):
        for i in range(j+1, len(array)):
            if abs(array[j] - array[i]) < 10**-7:
                print(f"{j}--{i} \n {array[j]}---{array[i]} \n Such a small probability")
                return False
    print('All right')
    return True


def data_generation(function, interval):
    points = np.linspace(interval[0], interval[1], 20)
    np.random.seed(10)
    f_x = [function(x) + np.random.choice([i for i in np.arange(-0.1, 0.1, 0.001) if abs(i) > 10**-7])
           for x in points for i in range(3)]
    # difference(f_x)
    points = points.tolist()
    points = points*3
    points = np.sort(points)
    return points, f_x


def Polinom(x, A_array):
    p_x = 0
    for i in range(len(A_array)):
        p_x += A_array[i] * x**i
    return p_x

def Get_summ(x_array, n, m):
    summ = 0
    for i in range(m):
        summ+= x_array[i]**n
    return summ

def Get_fX_summ(x_array, fx_array, n, m):
    summ = 0
    for i in range(m):
        summ+= fx_array[i]*(x_array[i]**n)
    return summ


def normal_equatiion(function, interval, n):
    x_n_array, m_values = data_generation(function, interval)

    # Vandermond's matrix   ------ there is error
    # E = [[x_n_array[i]**j if j != 0 else 1 for j in range(n+1)] for i in range(int(len(m_values)))]
    # A = matrix_mult(mtrx_T(E), E)
    # b = matrix_mult(mtrx_T(E), m_values)

    # Another way
    A = zero_mtrx(n+1)
    b = [0]*(n+1)
    for i in range(n+1):
        b[i] = Get_fX_summ(x_n_array, m_values, i, len(x_n_array))
        for j in range(n+1):
            if i == j == 0:
                A[i][j] = len(x_n_array)
            A[i][j] = Get_summ(x_n_array, j+i, len(x_n_array))
    A_array = np.linalg.solve(A,b)
    return A_array


def min_square_diff(interval, function, n):
    x = data_generation(function, interval)[0]
    m = len(x)
    summ = 0
    A_array = normal_equatiion(function, interval, n)
    for i in range(m):
        summ += (function(x[i]) - Polinom(x[i],A_array))**2
        # print(f"f_x: {function(x[i])} ----- pol_x: {Polinom(x[i], A_array)} ------- diff: {(function(x[i]) - Polinom(x[i],A_array))**2}")
    return summ

def Get_graph(function, interval, flag):
    x_values, fx_values = data_generation(function, interval)
    f_x = [function(x) for x in x_values]

    if flag == 1:
        N1_Pol = [Polinom(x, normal_equatiion(function, interval, 1)) for x in x_values]
        plt.title('n = 1')
        plt.scatter(x_values, fx_values)
        plt.plot(x_values, f_x, 'k',
                 x_values, N1_Pol, 'r')
        plt.show()
    elif flag == 2:
        N2_Pol = [Polinom(x, normal_equatiion(function, interval, 2)) for x in x_values]
        plt.scatter(x_values, fx_values)
        plt.title('n = 2')
        plt.plot(x_values, f_x, 'k',
                 x_values, N2_Pol, 'r')
        plt.show()
    elif flag == 3:
        N3_Pol = [Polinom(x, normal_equatiion(function, interval, 3)) for x in x_values]
        plt.scatter(x_values, fx_values)
        plt.title('n = 3')
        plt.plot(x_values, f_x, 'k',
                 x_values, N3_Pol, 'r')
        plt.show()
    elif flag == 4:
        N4_Pol = [Polinom(x, normal_equatiion(function, interval, 4)) for x in x_values]
        plt.scatter(x_values, fx_values)
        plt.title('n = 4')
        plt.plot(x_values, f_x, 'k',
                 x_values, N4_Pol, 'r')
        plt.show()
    elif flag == 5:
        N5_Pol = [Polinom(x, normal_equatiion(function, interval, 5)) for x in x_values]
        plt.scatter(x_values, fx_values)
        plt.title('n = 5')
        plt.plot(x_values, f_x, 'k',
                 x_values, N5_Pol, 'r')
        plt.show()


#
# for i in range(len(n)):
#     print(f" Min square difference, N = {n[i]}: {min_square_diff(interval, function, n[i])}")


#Ort polinoms

def Get_alfa_coeff(x_array, g_polinom):
    m = len(x_array)
    summ_1 = 0
    summ_2 = 0
    for i in range(m):
        summ_1 += x_array[i] * (g_polinom(x_array[i], x_array))**2
        summ_2 += g_polinom(x_array[i], x_array)**2
    alfa = summ_1 / summ_2
    return alfa

def Get_betta_coeff(x_array,g_pol, last_g_pol):
    m = len(x_array)
    summ_1 = 0
    summ_2 = 0
    if last_g_pol == 1:
        for i in range(m):
            summ_1 += x_array[i] * g_pol(x_array[i], x_array) * last_g_pol
            summ_2 += last_g_pol** 2
    else:
        for i in range(m):
            summ_1 += x_array[i] * g_pol(x_array[i], x_array) * last_g_pol(x_array[i], x_array)
            summ_2 += last_g_pol(x_array[i], x_array)**2
    betta = summ_1 / summ_2
    return betta

def Get_a_coeff(g_pol, x_array, fx_array):
    m = len(x_array)
    summ_1 = 0
    summ_2 = 0
    for i in range(m):
        if g_pol == 1:
            summ_1 += fx_array[i]
            summ_2 += g_pol
        else:
            summ_1 += g_pol(x_array[i], x_array)*fx_array[i]
            summ_2 += g_pol(x_array[i], x_array)**2
    a = summ_1 / summ_2
    return a

def ort_polinoms(x_, function, interval, n):
    x_array, fx_array = data_generation(function, interval)
    # x_array = [0, 1/4, 1/2, 3/4, 1]
    # fx_array = [1,2,1,0,1]
    m = len(x_array)

    g_old = 1

    def g(x, x_array):
        m = len(x_array)
        return x - 1/m * Get_summ(x_array, 1, m)

    g_array = [g_old, g]

    a_array = [Get_a_coeff(g_array[0], x_array, fx_array), Get_a_coeff(g_array[1], x_array, fx_array)]
    for j in range(1, n):

        alfa_j = Get_alfa_coeff(x_array, g_array[j])
        betta_j = Get_betta_coeff(x_array, g_array[j], g_array[j-1])

        # print(f"alfa: {alfa_j}")
        # print(f"betta: {betta_j}")

        def g_temp(x, x_array, g = g_array[j], g_old= g_array[j-1], alfa=alfa_j, betta=betta_j):
            if g_old == 1:
                return x * g(x, x_array) - alfa * g(x, x_array) - betta * g_old
            else:
                return x * g(x, x_array) - alfa * g(x, x_array) - betta * g_old(x, x_array)

        g_array = np.append(g_array, g_temp)


        a_array = np.append(a_array, Get_a_coeff(g_array[j+1], x_array, fx_array))


    result = 0


    for i in range(len(a_array)):
        if g_array[i] == 1:
            result += a_array[i]
        else:
            result += a_array[i]*g_array[i](x_, x_array)
    return result

# print(ort_polinoms(1, function, interval, 2))


def min_square_diff_ort(interval, function, n):
    x = data_generation(function, interval)[0]
    m = len(x)
    summ = 0
    for i in range(m):
        summ += (function(x[i]) - ort_polinoms(x[i], function, interval, n))**2
        # print(f"{x[i]} -- f_x: {function(x[i])} ----- pol_x: {ort_polinoms(x[i], function, interval, n)} ------- diff: {(function(x[i]) - ort_polinoms(x[i], function, interval, n))**2}")
    return summ

def Get_graph_ort(function, interval, flag):
    x_values, fx_values = data_generation(function, interval)
    f_x = [function(x) for x in x_values]


    if flag == 1:
        N1_Pol = [ort_polinoms(x, function, interval, 1) for x in x_values]
        N1_Pol_norm = [Polinom(x, normal_equatiion(function, interval, 1)) for x in x_values]
        plt.title('n = 1')
        plt.scatter(x_values, fx_values)
        plt.plot(x_values, f_x, 'k-',
                 x_values, N1_Pol, 'r-',
                 x_values, N1_Pol_norm,'b')
        plt.show()

    elif flag == 2:
        N2_Pol = [ort_polinoms(x, function, interval, 2) for x in x_values]
        N2_Pol_norm = [Polinom(x, normal_equatiion(function, interval, 2)) for x in x_values]
        plt.scatter(x_values, fx_values)
        plt.title('n = 2')
        plt.plot(x_values, f_x, 'k-',
                 x_values, N2_Pol, 'r-',
                 x_values, N2_Pol_norm, 'b')
        plt.show()
    elif flag == 3:
        N3_Pol = [ort_polinoms(x, function, interval, 3) for x in x_values]
        N3_Pol_norm = [Polinom(x, normal_equatiion(function, interval, 3)) for x in x_values]
        plt.scatter(x_values, fx_values)
        plt.title('n = 3')
        plt.plot(x_values, f_x, 'k-',
                 x_values, N3_Pol, 'r-',
                 x_values, N3_Pol_norm, 'b')
        plt.show()
    elif flag == 4:
        N4_Pol = [ort_polinoms(x, function, interval, 4) for x in x_values]
        N4_Pol_norm = [Polinom(x, normal_equatiion(function, interval, 4)) for x in x_values]
        plt.scatter(x_values, fx_values)
        plt.plot(x_values, f_x, 'k-',
                 x_values, N4_Pol, 'r-',
                 x_values, N4_Pol_norm, 'b')
        plt.show()
    elif flag == 5:
        N5_Pol = [ort_polinoms(x, function, interval, 5) for x in x_values]
        N5_Pol_norm = [Polinom(x, normal_equatiion(function, interval, 5)) for x in x_values]
        plt.scatter(x_values, fx_values)
        plt.title('n = 5')
        plt.plot(x_values, f_x, 'k-',
                 x_values, N5_Pol, 'r-',
                 x_values, N5_Pol_norm, 'b')
        plt.show()


for i in range(len(n)):
    Get_graph_ort(function, interval, n[i])

print()
# for i in range(len(n)):
#     print(f" Min square difference, N = {n[i]}: {min_square_diff_ort(interval, function, n[i])}")
for i in range(len(n)):
    # Get_graph(function, interval, n[i])
    Get_graph_ort(function, interval, n[i])