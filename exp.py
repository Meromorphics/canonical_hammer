import numpy
import scipy
import tensorflow
import torch
import pyexpokit

# todo: implement sparse arrays "natively" (ie, don't call
# scipy.sparse.csr_matrix) and try out different sparse matrix formats
def expA_v(A, v, factor, method, traceA=None):
    if method == "expm":
        return scipy.linalg.expm(factor*A) @ v
    elif method == "he_diag":
        W, V = scipy.linalg.eigh(A)
        return V @ numpy.diag(numpy.exp(factor*W)) @ V.T @ v
    elif method == "diag":
        W, V = scipy.linalg.eig(A)
        return V @ numpy.diag(numpy.exp(factor*W)) @ V.T @ v
    elif method == "expm_multiply":
        return scipy.sparse.linalg.expm_multiply(scipy.sparse.csr_matrix(factor*A), v, traceA=factor*traceA)
    elif method == "expm_tensor":
        return tensorflow.linalg.expm(factor*A).numpy() @ v
    elif method == "expm_torch":
        return torch.linalg.matrix_exp(torch.from_numpy(factor*A)).numpy() @ v
    elif method == "pyexpokit":
        return pyexpokit.expmv(factor, scipy.sparse.csr_matrix(A), v)