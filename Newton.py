import Functions as fn
import numpy as np
from QR_decomposition import SLAE as SLAE_QR


def f_test(x):
    return x - np.sin(x) - 0.25
def derv_f_test(x):
    return 1-np.cos(x)

def f(x):
    return x**3 - np.exp(x) + 1
def derv_f(x):
    return 3*x**2 - np.exp(x)
eps = 10**-4


def localization(function,n = 1, border_a = -100,border_b = 100):
    N = n # Стартовое
    result = [0,0]
    flag = False
    while flag != True:
        range_X_k = []
        for j in range(N + 1):
            range_X_k.append(border_a + j*(border_b - border_a)/N)        #Разбиваем на N частей
            # print(range_X_k)
        for i in range(len(range_X_k)-1):
            if function(range_X_k[i])*function(range_X_k[i+1]) < 0:          #Смотрим на значение функции
                result[0], result[1] = range_X_k[i], range_X_k[i+1]
                flag = True
        if flag == False:
            N = N*2
    print(result)
    return result

def replace_boards(f, range_,c):
    if fn.sgn(f(range_[0])) == fn.sgn(f(c)):      # Сравниваем знаки на промежутках и меняем границу
        range_[0] = c
    else:
        range_[1] = c
    return range_

def Newton(f, derv_f):
    range_ = localization(f)
    x_n = range_[1]
    x_old = 0
    while abs(x_n - x_old) > eps:
        x_old = x_n
        x_n = x_old - f(x_old)/derv_f(x_old)
        if (x_n <= range_[1]) and (x_n >= range_[0]):
            range_ = replace_boards(f,range_, x_n)
        else:
            print('Yeeea')
            x_n = range_[0] - (range_[1] - range_[0])/(f(range_[1]) - f(range_[0]))*f(range_[0])        # Метод хорд
            # x_n = (range_[0] + range_[1])/2      # Метод половинного деления
            range_ = replace_boards(f,range_, x_n)
    return x_n

print(Newton(f, derv_f))
result = Newton(f,derv_f)
print(f(result))

# print(Newton(f_test, derv_f_test))
# result_test = Newton(f_test, derv_f_test)
# print(f_test(result_test))

def F1(x,y, lb = 1):
    return lb*np.sin(y) + 2*x - 2
def F2(x,y, lb = 1):
    return y + lb*np.cos(x-1) - 0.7
def df1_dx(x,lb = 1):
    return 2
def df1_dy(y, lb = 1):
    return lb*np.cos(y)
def df2_dx(x, lb = 1):
    return lb*(-np.sin(x-1))
def df2_dy(y, lb = 1):
    return 1

def F1_test(x,y, lb = 1):
    return lb*np.sin(y+2) - x - 1.5
def F2_test(x,y, lb =1 ):
    return y + lb*np.cos(x-2) - 0.5
def df1_test_dx(x, lb = 1):
    return -1
def df1_test_dy(y, lb = 1):
    return lb*np.cos(y+2)
def df2_test_dx(x, lb = 1):
    return lb*(-np.sin(x-2))
def df2_test_dy(y, lb = 1):
    return 1


def getStartValue(df1_dx,df1_dy, df2_dx, df2_dy,F1,F2):
    x_0 = 1             # Вручную посчитанное 0-е начальное приближение
    y_0 = 0.7
    x_k = [x_0, y_0]
    N = 10
    for i in range(1, N):
        x_old = x_k
        F = [F1(x_old[0], x_old[1], i/N), F2(x_old[0], x_old[1], i/N)]
        J_M = [[df1_dx(x_old[0], i/N), df1_dy(x_old[1], i/N)], [df2_dx(x_old[0], i/N), df2_dy(x_old[1],i/N)]]
        dlt_x = SLAE_QR(J_M, fn.sk_operation(F, -1, 'mult'))
        x_k = fn.matrix_operation(x_old, dlt_x, 'add')
        # print(i, x_k)
    return x_k



def Newton_system(df1_dx,df1_dy,df2_dx,df2_dy,F1,F2):
    # x_0 = 2.0
    # y_0 = -0.9 # f - По графику
    # x_0 = -2.5  #f_test - По графику
    # y_0 = 1.5
    # x_old = [x_0, y_0]
    x_old = getStartValue(df1_dx, df1_dy, df2_dx, df2_dy, F1, F2)
    # print(x_old)
    J_M = [[df1_dx(x_old[0]), df1_dy(x_old[1])] ,[df2_dx(x_old[0]), df2_dy(x_old[1])]]
    F = [F1(x_old[0], x_old[1]), F2(x_old[0], x_old[1])]

    dlt_x = SLAE_QR(J_M, fn.sk_operation(F, -1, 'mult'))
    x_k = fn.matrix_operation(x_old, dlt_x, 'add')

    while(fn.norma_v(fn.matrix_operation(x_k, x_old, 'diff'), 0)) > eps:
        x_old = x_k
        J_M = [[df1_dx(x_old[0]), df1_dy(x_old[1])], [df2_dx(x_old[0]), df2_dy(x_old[1])]]
        F = [F1(x_old[0], x_old[1]), F2(x_old[0], x_old[1])]
        dlt_x = SLAE_QR(J_M, fn.sk_operation(F, -1, 'mult'))
        x_k = fn.matrix_operation(x_old, dlt_x, 'add')
    return x_k
print(145%24)



# print(Newton_system(df1_dx,df1_dy,df2_dx,df2_dy,F1,F2))
# print(Newton_system(df1_test_dx,df1_test_dy,df2_test_dx,df2_test_dy,F1_test,F2_test))
# result = Newton_system(df1_dx, df1_dy, df2_dx, df2_dy,F1,F2)
# print(F1(result[0], result[1]))
# result2 = Newton_system(df1_test_dx, df1_test_dy, df2_test_dx, df2_test_dy, F1_test, F2_test)
# print(F1_test(result2[0], result2[1]))
