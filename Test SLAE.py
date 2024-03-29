import Functions as fn
import numpy as np
import pandas as pd
import warnings
from Simple_iters import SLAE as SLAE_Si
from QR_decomposition import SLAE as SLAE_QR
from LU_decomposition import SLAE as SLAE_LU

from Zeidel import  SLAE as SLAE_Z
warnings.filterwarnings('ignore')
N = 2
TEST_A=[[[0,2,3],[1,2,4],[4,5,6]],      #test0
       [[N+2,1,1],[1,N+4,1],[1,1,N+6]], #test1
       [[-(N+2),1,1],[1,-(N+4),1],[1,1,-(N+6)]],#test2
       [[-(N+2),N+3,N+4],[N+5,-(N+4),N+1],[N+4,N+5,-(N+6)]],#test3
       [[N+2,N+1,N+1],[N+1,N+4,N+1],[N+1,N+1,N+6]]] #test4

TEST_B=[[13,17,32],#test0
        [N+4,N+6,N+8],#test1
        [-(N+4), -(N+6), -(N+8)],#test2
        [N+4,N+6, N+8],#test3
        [N+4,N+6,N+8]]#test4

def Bad_SLAE(n): #test 5
    N = 2
    eps = [10**-3, 10**-4, 10**-6]
    A = fn.matrix_operation([[-1 if j >= i else 0 for j in range(n)] for i in range(n)], fn.sk_operation(fn.unit_mtrx(n), 2, 'mult'), 'add')
    A_ = fn.sk_operation([[-1 if j > i else 1 for j in range(n)] for i in range(n)], N*eps[0], 'mult')
    b = [-1 for i in range(n)]
    b[len(b) -1] = 1
    slae = fn.matrix_operation(A,A_, 'add')
    return slae, b

def print_Si(epsilon, n):
    result = [0] * 5
    result_np = [0]*5
    result_eps = [0]*5
    for i in range(5):
        result[i] = SLAE_Si(TEST_A[i],TEST_B[i], epsilon)
        result_np[i] = np.linalg.solve(TEST_A[i],TEST_B[i])
        result_eps[i] = np.abs(fn.matrix_operation((result[i]), result_np[i], 'diff'))
    print('My')
    print(np.array(result))
    print('NP')
    print(np.array(result_np))
    print("epsilon")
    print(np.array(result_eps))
    print('Bad SlAE - my')
    A, b = Bad_SLAE(n)
    x = SLAE_Si(A, b, epsilon)
    print(np.array(x))
    print('BAD SLAE - numpy')
    x_np = np.linalg.solve(A, b)
    print(x_np)
    print('epsilon BAD SLAE')
    print(np.abs(fn.matrix_operation(x_np, x, 'diff')))


def print_Z(epsilon, n):
    result = [0] * 5
    result_np = [0]*5
    result_eps = [0]*5
    for i in range(5):
        result[i] = SLAE_Z(TEST_A[i],TEST_B[i], epsilon)
        result_np[i] = np.linalg.solve(TEST_A[i],TEST_B[i])
        result_eps[i] = np.abs(fn.matrix_operation((result[i]), result_np[i], 'diff'))
    print('My')
    print(np.array(result))
    print('NP')
    print(np.array(result_np))
    print("epsilon")
    print(np.array(result_eps))
    print('Bad SlAE - my')
    A, b = Bad_SLAE(n)
    x = SLAE_Z(A, b, epsilon)
    print(np.array(x))
    print('BAD SLAE - numpy')
    x_np = np.linalg.solve(A, b)
    print(x_np)
    print('epsilon BAD SLAE')
    print(np.abs(fn.matrix_operation(x_np, x, 'diff')))

def print_LU(n):
    result = [0] * 5
    result_np = [0]*5
    result_eps = [0]*5
    for i in range(5):
        result[i] = SLAE_LU(TEST_A[i],TEST_B[i])
        result_np[i] = np.linalg.solve(TEST_A[i],TEST_B[i])
        result_eps[i] = np.abs(fn.matrix_operation((result[i]), result_np[i], 'diff'))
    print('My')
    print(np.array(result))
    print('NP')
    print(np.array(result_np))
    print("epsilon")
    print(np.array(result_eps))
    print('Bad SlAE - my')
    A, b = Bad_SLAE(n)
    x = SLAE_LU(A, b)
    print(np.array(x))
    print('BAD SLAE - numpy')
    x_np = np.linalg.solve(A, b)
    print(x_np)
    print('epsilon BAD SLAE')
    print(np.abs(fn.matrix_operation(x_np, x, 'diff')))

def print_QR(n):
    result = [0] * 5
    result_np = [0]*5
    result_eps = [0]*5
    for i in range(5):
        result[i] = SLAE_QR(TEST_A[i],TEST_B[i])
        result_np[i] = np.linalg.solve(TEST_A[i],TEST_B[i])
        result_eps[i] = np.abs(fn.matrix_operation((result[i]), result_np[i], 'diff'))
    print('My')
    print(np.array(result))
    print('NP')
    print(np.array(result_np))
    print("epsilon")
    print(np.array(result_eps))
    print('Bad SlAE - my')
    A,b = Bad_SLAE(n)
    x = SLAE_QR(A, b)
    print(np.array(x))
    print('BAD SLAE - numpy')
    x_np = np.linalg.solve(A, b)
    print(x_np)
    print('epsilon BAD SLAE')
    print(np.abs(fn.matrix_operation(x_np, x,'diff')))

# print_Z(10**-6,6)
print_Si(10**-3,10)
# print_LU(10)
# print_QR(10)

