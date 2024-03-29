import Functions as fn
import numpy as np

def get_B(A,b):
    n = len(A)
    mu = 1/fn.norm_matrix(A, 0)   # Норма А ?
    c = fn.sk_operation(b, mu, 'mult')
    B = fn.matrix_operation(fn.unit_mtrx(n), fn.sk_operation(A,mu,'mult'), 'diff')
    return B


def get_norm(B):
    norms = [fn.norm_matrix(B,i) for i in range(2)]
    flag = False
    ind = 0
    for i in range(2):
        if norms[i] < 1:
            flag = True
            ind = i
            return ind
        else:
            flag = False
    if flag == True:
        return ind
    else:
        return 'False'

def Alg_simple_iters(B,norm_id,b,epsilon,A):
    c = x_old = fn.sk_operation(b, 1 / fn.norm_matrix(A, 0), 'mult')
    x_k = fn.matrix_operation(fn.matrix_mult(B, x_old), c, 'add')
    while fn.norm_matrix(B, norm_id) / (1 - fn.norm_matrix(B, norm_id)) * fn.norma_v(fn.matrix_operation(x_k, x_old, 'diff'), 0) > epsilon:
        x_old = x_k
        x_k = fn.matrix_operation(fn.matrix_mult(B, x_old), c, 'add')
    return x_k




def SLAE(A,b, epsilon):
    B = get_B(A, b)
    norm_id = get_norm(B)
    x_k =[]
    if norm_id == 'False':
        A_temp = fn.matrix_mult(fn.mtrx_T(A), A)
        b_temp = fn.matrix_mult(fn.mtrx_T(A), b)
        B = get_B(A_temp, b_temp)
        norm_id = get_norm(B)
        if norm_id == 'False':
            c = x_old = fn.sk_operation(b_temp, 1 / fn.norm_matrix(A_temp, 0), 'mult')
            x_k = fn.matrix_operation(fn.matrix_mult(B, x_old), c, 'add')
            while fn.norma_v(fn.matrix_operation(fn.matrix_mult(A_temp, x_k), b_temp, 'diff'), 1) >= epsilon:    # Другой критерий остановки
                x_old = x_k
                x_k = fn.matrix_operation(fn.matrix_mult(B, x_old), c, 'add')
        else:
            x_k = Alg_simple_iters(B, norm_id, b_temp, epsilon, A_temp)
    else:
        x_k = Alg_simple_iters(B, norm_id, b, epsilon, A)
    return x_k





A = [[3,1,1],
     [1,5,1],
     [1,1,7]]

b = [5,7,9]
N =2

TEST_A=[[[0,2,3],[1,2,4],[4,5,6]],      #test0
       [[N+2,1,1],[1,N+4,1],[1,1,N+6]], #test1
       [[-(N+2),1,1],[1,-(N+4),1],[1,1,-(N+6)]],#test2
       [[-(N+2),N+3,N+4],[N+5,-(N+4),N+1],[N+4,N+5,-(N+6)]],#test3
       [[N+2,N+1,N+1],[N+1,N+4,N+1],[N+1,N+1,N+6]]] #test4

TEST_B=[[13,17,32],#test0
        [N+4,N+6,N+8],#test1
        [-(N+4),-(N+6),-(N+8)],#test2
        [N+4,N+6,N+8],#test3
        [N+4,N+6,N+8]]#test4

# x = SLAE(TEST_A[2],TEST_B[2],10**-3)
# x_ans = np.linalg.solve(TEST_A[2], TEST_B[2])
# print('NumPy')
# print(x_ans)
# print('My')
# print(x)
# print("Eps")
# eps = [abs(fn.matrix_operation(x_ans,x,'diff')[i]) for i in range(len(x_ans))]
# print(eps)


