from copy import deepcopy
import numpy as np
import random
def sgn(x):
    if x >= 0:
        return 1
    else:
        return -1

def matrix_mult(first, second):
    if isinstance(first[0],list)&(isinstance(second[0],list)):  # Матричное умножение
        if len(first[0]) != len(second):
            return "Error dim"
        length = len(first)
        result = []
        for i in range(length):
            temp_result = []
            for n in range(length):
                element = 0
                for j in range(length):
                    element += first[i][j]*second[j][n]
                temp_result.append(element)
            result.append(temp_result)
        return result

    elif isinstance(first[0], list)&(isinstance(second, list)):  # Матрица на вектор
        n = len(first)
        ans =[]
        for i in range(n):
            summ = 0
            for j in range(n):
                summ += first[i][j] * second[j]
            ans.append(summ)
        return ans
    elif isinstance(first, list)&(isinstance(second[0],list)):      # вектор на матрицу
        n = len(first)
        ans = []
        for i in range(n):
            summ = 0
            for j in range(n):
                summ += second[i][j] * first[j]
            ans.append(summ)
        return ans

def w_WT(w1_,w2_):     #Траспонирование вектора
    w1 = deepcopy(w1_)
    w2 = deepcopy(w2_)
    Ans = []
    n = len(w1)
    for i in range(n):
        zero_str = [0]*n
        Ans.append(zero_str)
        for j in range(n):
            Ans[i][j] = w1[i]*w2[j]
        zero_str = zero_str[:]
    return Ans

def sk_operation(A_,c,flag):     #операции со скалярами(матрица/вектор, скаляр, операция)
    A = deepcopy(A_)
    n = len(A_)
    if isinstance(A[0], list):# Матрица на скаляр
        if flag == 'mult':
            for i in range(n):
                for j in range(n):
                    A[i][j] *= c
            return A
        elif flag == 'div':
            if c == 0:
                c = 10**-15
            for i in range(n):
                for j in range(n):
                    A[i][j] /= c
            return A
        else:
            return "Errror operations"
    elif isinstance(A[0], list) == False:  # Скаляр на вектор
        if flag == 'mult':
            for i in range(n):
                A[i] *= c
            return A
        if flag == 'div':
            if c == 0:
                c = 10**-15
            for i in range(n):
                A[i] /= c
            return A
        else:
            return "Errror operations"

def matrix_operation(A1_, A2_, flag):   # Сложение и разность матриц
    A1 = deepcopy(A1_)
    A2 = deepcopy(A2_)
    n = len(A1_)
    if isinstance(A1[0], list) & (isinstance(A2[0], list)):
        if flag == 'add':
            Ans = []
            for i in range(n):
                zero_str = [0] * n
                Ans.append(zero_str)
                for j in range(n):
                    c = A1[i][j] + A2[i][j]
                    Ans[i][j] = c
                zero_str = zero_str[:]
            return Ans
        elif flag == 'diff':
            Ans = []
            for i in range(n):
                zero_str = [0] * n
                Ans.append(zero_str)
                for j in range(n):
                    c = A1[i][j] - A2[i][j]
                    Ans[i][j] = c
                zero_str = zero_str[:]
            return Ans
        else:
            return "Error operation"
    elif (isinstance(A1[0], list)&isinstance(A2[0], list)) == False:
        if flag == 'add':
            for i in range(n):
                A1[i] += A2[i]
            return A1
        elif flag == 'diff':
            for i in range(n):
                A1[i] -= A2[i]
            return A1
        else:
            return "Error operation"
    else:
        return "Vector and Matrix"
    
def SLAE_low_triangle_mtrx(L,b):    # Нижнетреугольная матрица
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

def SLAE_hight_triangle_mtrx(U,y): # Верхнетреугольная матрица
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

def id_el_max(M,i):   # id max элемента
    n = len(M)
    max = 0
    id_max = 0
    for j in range(i, n):
        if abs(M[j][i]) > max:
            max = abs(M[j][i])
            id_max = j
    return id_max

def zero_mtrx(n): #Нулевая матрица
    P = []
    for i in range(n):
        zero_str = [0] * n
        P.append(zero_str)
        P[i][i] = 0
        zero_str = zero_str[:]
    return P

def unit_mtrx(n): # Единичная матрица
    P = []
    for i in range(n):
        zero_str = [0]*n
        P.append(zero_str)
        P[i][i] = 1
        zero_str = zero_str[:]
    return P

def replace_str(M_, i,j):  # замена строчек в матрице
    M = deepcopy(M_)
    temp = []
    temp = M[i]
    M[i] = M[j]
    M[j] = temp
    return M

def norma_v(v_, i):  # Норма вектора
    if i == 0:     #Евклидова
        n = len(v_)
        summ = 0
        for i in range(n):
            summ += v_[i]**2
        return np.sqrt(summ)
    elif i == 1:      #Масимальный модуль
        v = deepcopy(v_)
        for i in range(len(v)):
            v[i] = abs(v[i])
        return max(v)


def norm_matrix(A_,i):   # Норма матрицы
    A = deepcopy(A_)
    n = len(A)
    result = 0
    if i == 0:
        for i in range(n):
            A[i] = [abs(A[i][j]) for j in range(len(A[i]))]
            if sum(A[i]) > result:
                result = sum(A[i])
        return result
    elif i == 1:
        A = mtrx_T(A)
        for i in range(n):
            A[i] = [abs(A[i][j]) for j in range(len(A[i]))]
            if sum(A[i]) > result:
                result = sum(A[i])
        return result

def mtrx_T(A_):  # Траспонирование матрицы
    A = deepcopy(A_)
    if isinstance(A[0], list):
        n = len(A)
        n_new = len(A[0])
        result = []
        for i in range(n_new):
            zero_str = [0]*n
            result.append(zero_str)
            for j in range(n):
                result[i][j] = A[j][i]
            zero_str = zero_str[:]
        return result
    else:
        return A


def Print(A):
    A = np.array(A)
    return print(A)


def SLAE_Test(n,SLAE):
    A = [[random.randint(1, 11) for j in range(n)] for i in range(n)]
    b = [random.randint(1, 11) for l in range(n)]
    # print('M= ', A)
    # print('b= ',b)
    x = SLAE(A, b)
    epsilon = []
    # print('My= ', x)
    # print()
    x_ans = np.linalg.solve(A, b)
    # print('NumPy= ', x)
    for i in range(len(x)):
        epsilon.append(x[i] - x_ans[i])
    print('Eps= ', epsilon)



def skalar_mult_v(v1,v2):
        result = 0
        n = len(v1)
        for i in range(n):
            result += v1[i] * v2[i]
        return result

