import Functions as fn
from copy import deepcopy
import random
import numpy as np



def create_C_d(A,b):
    R = deepcopy(A)
    d = deepcopy(b)
    C = []
    n = len(A)
    d = [0]*n
    for i in range(n):
        zero_str = [0]*n
        if R[i][i] == 0:
            d[i] = b[i] / 10**-15
        else:
            d[i] = b[i]/R[i][i]
        C.append(zero_str)
        for j in range(n):
            if i == j:
                C[i][j] = 0
            else:
                if R[i][i] == 0:
                    d[i] = b[i] / 10 ** -15
                else:
                    C[i][j] = -R[i][j]/R[i][i]
        zero_str = zero_str[:]
    return C, d

def get_C_x(C,x_):          # Коэффициенты матрицы
    n = len(C)
    x = deepcopy(x_)
    x_i = []
    C_i = deepcopy(C)
    for i in range(n):
        x.pop(i)
        x_i.append(x)
        x = deepcopy(x_)
        for j in range(i,i+1):
            C_i[i].pop(j)
    return C_i, x_i



def SLAE_i(C, x_k, d):       #Считает следующий вектор
    n = len(C)
    x_temp = deepcopy(x_k)
    C_i, x_k = get_C_x(C, x_temp)  #Коэфф-ты
    x1 = [0]*n
    for i in range(n):
        x1[i] = fn.skalar_mult_v(C_i[i], x_k[i]) + d[i]
        x_temp[i] = x1[i]
        x_k = get_C_x(C, x_temp)[1]
    return x1

def SLAE_Work(C,d,b, x_k, epsilon, norma,A):
    while norma > epsilon:
        x_k = SLAE_i(C, x_k, d)
        norma = fn.norma_v(fn.matrix_operation(fn.matrix_mult(A, x_k), b, 'diff'), 1)
    return x_k

def check_daig(A_,b):
    A = deepcopy(A_)
    for i in range(len(A)):
        summ = 0
        A_i_i = abs(A[i][0])
        for j in range(len(A)):
            if j != i:
                summ += abs(A[i][j])
        if A_i_i > summ:
            continue
        else:
            return fn.matrix_mult(fn.mtrx_T(A_), A_),fn.matrix_mult(fn.mtrx_T(A_), b)
    return A_, b


def SLAE(A,b,epsilon):
    A, b = check_daig(A,b)
    C,d = create_C_d(A,b)
    x_k = d
    x_k = SLAE_i(C, x_k, d)
    norma = fn.norma_v(fn.matrix_operation(fn.matrix_mult(A, x_k), b, 'diff'),1)
    x_k = SLAE_Work(C, d, b, x_k, epsilon, norma,A)
    return x_k



# A = [ [4, 2, 3,1],
#       [1, 5, 4,3],
#       [4, 5, 6,2],
#       [3, 1, 6,9]]
# print(A)
# b = [13,17,32,71]
#
#
# x_ans = np.linalg.solve(A, b)
# x = SLAE(A, b,10**-3)
# print(x)
# print()
# print(x_ans)

