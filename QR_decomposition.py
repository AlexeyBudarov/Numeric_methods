import Functions as fn
import numpy as np
import random
from copy import deepcopy


def get_W(R_,i):                 # Получаем вектор w
    R = deepcopy(R_)
    n = len(R)
    z = fn.unit_mtrx(n)[0]
    if i == 0:
        R = R[i:]
        y = fn.mtrx_T(R)[i] # y = Первый столбец R
        z = z[:len(z) - i]
    else:
        R = R[1:]
        y = fn.mtrx_T(R)[1] # В остальных случаях - срезаем первую строчку - транспонируем и берем 2ю строчку = y
        z = z[:len(z) - 1]
    alf = fn.norma_v(y, 0)
    p = fn.norma_v(fn.matrix_operation(y, fn.sk_operation(z, alf, 'mult'), 'diff'), 0)
    W = fn.sk_operation(fn.matrix_operation(y, fn.sk_operation(z, alf, 'mult'), 'diff'), p, 'div')
    return W

def Q_i_mult(Q_result):
    Q_result = Q_result[::-1]
    ans_Q = Q_result[0]
    for i in range(len(Q_result) - 1):  # Перемножение Q матриц
        Q_result[i] = ans_Q
        ans_Q = fn.matrix_mult(ans_Q, Q_result[i + 1])
    return fn.mtrx_T(ans_Q)

def R_slice(R_,i):
    R_ = R_[i:]
    R_ = fn.mtrx_T(R_)
    R_ = R_[i:]
    R_ = fn.mtrx_T(R_)
    return R_

def reflect_mtrx(W):
    n = len(W)
    Q = fn.matrix_operation(fn.unit_mtrx(n), fn.w_WT(fn.sk_operation(W, 2, 'mult'), W), 'diff')
    return Q

def transform_mtrx(Q,R,i):
    if i == 0:
        return fn.matrix_mult(Q, R)
    R = R_slice(R, 1)
    return fn.matrix_mult(Q, R)

def concatenate_R(R, R_old):
    k = len(R_old) - len(R)
    n = len(R_old)
    for i in range(k,n):
        for j in range(k, n):
            R_old[i][j] = R[i-k][j-k]
    return R_old

def concatenate_Q(Q, n):
    k = n - len(Q)
    Q_old = fn.unit_mtrx(n)
    for i in range(k,n):
        for j in range(k, n):
            Q_old[i][j] = Q[i-k][j-k]
    return Q_old

def QR_dec(A):
    n = len(A)
    R = deepcopy(A)
    R_old = []
    Q_result = []
    Q_i = fn.unit_mtrx(n)
    for i in range(n-1):
        W = get_W(R, i)
        if i < 2:               # На 0,1 добавляем Q0 и Q1
            Q_result.append(Q_i)
        Q_i = reflect_mtrx(W)                 # Изменяем Q
        if i <= 1:                            # Сохраняем копию, чтобы на 2м(1м) шаге получить R2
            R_old = deepcopy(R)
        R = transform_mtrx(Q_i, R, i)         # Дальше изменяется размерность, в transform_ срезаем строчки и столбцы
        if i != 0:
            Q_result.append(concatenate_Q(Q_i, len(A)))
            R_old = concatenate_R(R,R_old)

    ans_R = R_old
    ans_Q = Q_i_mult(Q_result)   # Перемножение Q матриц
    return ans_Q, ans_R

def SLAE(R,b):
    result = QR_dec(R)
    Q = result[0]
    R = result[1]
    y = fn.matrix_mult(fn.mtrx_T(Q), b)
    x = fn.SLAE_hight_triangle_mtrx(R, y)
    return x


# A = [[3,1,1,1],
#      [1,5,1,2],
#      [1,1,7,3],
#      [1,1,9,4]]
# b = [5,7,9,11]
# print(SLAE(A,b))

