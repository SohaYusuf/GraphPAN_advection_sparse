import torch
from scipy.sparse.linalg import lgmres
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator


def gmres_with_preconditioner(A, b, u_true, tol, plot, M, method, path):

    n = A.shape[0]
    print('type A: ', type(A))

    # csr_data = A.values().numpy()
    # csr_indices = A.col_indices().numpy()
    # csr_indptr = A.crow_indices().numpy()
    # A = csr_matrix((csr_data, csr_indices, csr_indptr), shape=A.shape)
    # b = b.numpy()
    # u_true = u_true.numpy()

    # indices = A.indices().numpy()
    # values = A.indices().numpy()
    # A = coo_matrix((values, (indices[0], indices[1])), shape=A.shape)
    #####################################################################
    A = A.to_dense().numpy()
    b = b.to_dense().numpy()
    u_true = u_true.to_dense().numpy()
    ####################################################################
    #A = A.to_dense().numpy()
    #A = coo_matrix(A)
    
    # global iteration
    # iteration = 0
    residuals = []
    
    # def callback(residual):
    #     global iteration
    #     #res = b - A @ xk
    #     residual_norm = np.linalg.norm(residual)
    #     print(f'Iteration: {iteration} ==========> Residual: {residual_norm}')
    #     if iteration >=1:
    #         residuals.append(residual_norm)
    #     iteration = iteration +1

    def callback(xk):
        callback.iterations += 1
        r = np.linalg.norm(b - A @ xk)
        residuals.append(r)
        print(f"Iteration {callback.iterations}, Residual: {r}")

    callback.iterations = 0

    ####################################################################################################

    def matvec(v):
        return A @ v

    A_op = LinearOperator((n, n), matvec=matvec)

    ####################################################################################################

    u_gmres, info = lgmres(A_op, b, M=M, atol=tol, callback=callback, maxiter=n)

    if info == 0:
        print("Converged to solution")
        print(u_gmres)
        print("Number of inner GMRES iterations:", callback.iterations)
    else:
        print("Convergence failed. Information code:", info)

    u_gmres = u_gmres.reshape(-1,1)
    print('norm of u_gmres: ',np.linalg.norm(u_gmres))
    u_true = u_true.reshape(-1,1)

    error = np.linalg.norm(u_gmres - u_true)
    residual_final = np.linalg.norm(b - A @ u_gmres)
    print('error |x_true - x_hat|: ',error)
    print('Final residual error: ',residual_final)
    iterations_ = callback.iterations

    if plot:
        print(f"Number of iterations for {method}:", iterations_)
        plt.figure(1)
        plt.plot(residuals, label=method)
        plt.title(f'GMRES for random non-symmetric data (n={n})')
        plt.xlabel('# iteration')
        plt.ylabel('solution norm')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{path}/{method}_gmres.png')
    
    return iterations_, residuals

def gmres_without_preconditioner(A, b, u_true, tol, plot, method, path):

    n = A.shape[0]
    
    # print('type A: ', type(A))
    # csr_data = A.values().numpy()
    # csr_indices = A.col_indices().numpy()
    # csr_indptr = A.crow_indices().numpy()
    # A = csr_matrix((csr_data, csr_indices, csr_indptr), shape=A.shape)
    # b = b.numpy()
    # u_true = u_true.numpy()
    
    # indices = A.indices().numpy()
    # values = A.indices().numpy()
    #A = A.to_dense().numpy()
    #A = coo_matrix(A)
    ##########################################################################
    A = A.to_dense().numpy()
    b = b.to_dense().numpy()
    u_true = u_true.to_dense().numpy()
    # error = A @ u_true - b
    ##############################################################################
    #print('error |x_true - x_hat|: ',error)
    
    # global iteration
    # iteration = 0
    residuals = []
    
    # def callback(residual):
    #     global iteration
    #     #res = b - A @ xk
    #     residual_norm = np.linalg.norm(residual)
    #     print(f'Iteration: {iteration} ==========> Residual: {residual_norm}')
    #     if iteration >=1:
    #         residuals.append(residual_norm)
    #     iteration = iteration +1

    def callback(xk):
        callback.iterations += 1
        r = np.linalg.norm(b - A @ xk)
        residuals.append(r)
        print(f"Iteration {callback.iterations}, Residual: {r}")

    callback.iterations = 0

    ####################################################################################################

    def matvec(v):
        return A @ v

    A_op = LinearOperator((n, n), matvec=matvec)

    ####################################################################################################
            
    u_gmres, info = lgmres(A_op, b, atol=tol, callback=callback, maxiter=n)

    if info == 0:
        print("Converged to solution")
        print(u_gmres)
        print("Number of inner GMRES iterations:", callback.iterations)
    else:
        print("Convergence failed. Information code:", info)

    u_gmres = u_gmres.reshape(-1,1)
    print('norm of u_gmres: ',np.linalg.norm(u_gmres))
    u_true = u_true.reshape(-1,1)

    error = np.linalg.norm(u_gmres - u_true)
    residual_final = np.linalg.norm(b - A @ u_gmres)
    print('error |x_true - x_hat|: ',error)
    print('Final residual error: ',residual_final)
    iterations_ = callback.iterations
    
    if plot:
        print(f"Number of iterations for {method}:", iterations_)
        plt.figure(1)
        plt.plot(residuals, label=method)
        plt.title(f'GMRES for random non-symmetric data (n={n})')
        plt.xlabel('# iteration')
        plt.ylabel('solution norm')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{path}/{method}_gmres.png')
    
    return iterations_, residuals