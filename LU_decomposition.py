import numpy as np
from copy import deepcopy
import Functions as fn
import random

def mtrx_transform(M_, i):
    M = deepcopy(M_)
    n = len(M)
    for j in range(i+1, n):
        M[j][i] = M[j][i]/M[i][i]
        for k in range(i+1, n):
            M[j][k] = M[j][k] - M[j][i]*M[i][k]
    return M



def low_triang_mtrx(M):
    M_temp = deepcopy(M)
    n = len(M)
    Ans = []
    for i in range(n):
        zero_str = [0] * n
        Ans.append(zero_str)
        for j in range(i, n):
            Ans[i][j] = M_temp[i][j]
        zero_str = zero_str[:]
    return fn.matrix_operation(fn.matrix_operation(M, Ans, 'diff'), fn.unit_mtrx(n), 'add')


def LU_dec(M):
    n = len(M)
    P = fn.unit_mtrx(n)
    for i in range(len(M)):
        j = fn.id_el_max(M, i)
        P = fn.replace_str(P, i, j)
        M = fn.replace_str(M, i, j)
        M = mtrx_transform(M, i)
    summ_L_U = fn.matrix_operation(M, fn.unit_mtrx(n), 'add')
    L = low_triang_mtrx(summ_L_U)
    U = fn.matrix_operation(summ_L_U, L, 'diff')
    return L, U, P

def Ly_b(L,b):
    n = len(L)
    y = []
    for i in range(n):
        x0 = 0
        for j in range(0, i+1):
            if i != j:
                x0+= L[i][j]*y[j]
            else:
                y.append((b[i]-x0)/L[i][j])
    return y

def Ux_y(U,y):
    n = len(U)
    x = []

    for i in range(n-1, -1, -1):
        y0 = 0
        for j in range(n-1, i-1, -1):
            if i != j:
                y0 += U[i][j] * x[n-j-1]
            else:
                x.append((y[i] - y0) / U[i][j])
    x.reverse()
    return x

def SLAE(A, b):
    result = LU_dec(A)
    B = fn.matrix_mult(result[2], b)
    n = len(A)
    return Ux_y(result[1], Ly_b(result[0], B))










# def det_(A): # Пока работает только с 3 на 3 матрицами
#     summ = 0
#     def one_minor(A, multiplier=1, el=0):
#         A_ = A.copy()
#         A_temp = list()
#         if len(A_) != len(A_[0]):
#             return "The matrix isn't square"
#         multiplier *= A_[0][el]
#         for i in range(len(A_) - 1):
#             A_temp.append(A_[i + 1:][0].copy())
#             A_temp[i].pop(el)
#         while len(A_temp) != 2:
#             return one_minor(A_temp, multiplier)
#         else:
#             return multiplier * (A_temp[0][0] * A_temp[1][1] - A_temp[0][1] * A_temp[1][0])
#
#     for n in range(len(A)):
#         if len(A) % 2 == 0:
#             summ += (-1)**(n+1)*one_minor(A, el = n)
#         else:
#             summ += (-1) **n * one_minor(A, el=n)
#     return summ





















