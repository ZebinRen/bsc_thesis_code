import tensorflow as tf
import numpy as np
import scipy.sparse as sp

def create_load_sparse(sparse_mat):
    '''
    The sparse matrix is saved as a scipy sparse matrix
    It cannot be directly used by TF
    This will change the sparse matrix to tuple representation
    index, values, shape
    '''
    if not sp.isspmatrix_coo(sparse_mat):
        sparse_mat = sparse_mat.tocoo()
    indexs = np.vstack((sparse_mat.row, sparse_mat.col)).transpose()
    data = sparse_mat.data
    shape = sparse_mat.shape

    #Type cast
    indexs = indexs.astype(np.int64)
    data = data.astype(np.float32)
    shape = np.array(list(shape))
    shape = shape.astype(np.int64)

    return (indexs, data, shape)

def symmetric_normalized_laplacian(adjancy):
    '''
    Given a Lapllacian Matrix
    Compute its symmetric normalized form
    Arguments:
    L: Laplacian Matrix
    is_sparse: If L is sparse
    Return:
    D^-0.5 L D^-0.5
    '''
    #convert to coo matrix for computation
    adjancy = sp.coo_matrix(adjancy)
    rowsum = np.array(adjancy.sum(1))

    #Compute D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    #If it is inf(The inverse of 0, then set it to 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized = adjancy.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    return normalized
    


def row_normalized(mat):
    '''
    row normalized feature
    '''
    #Compute row sum and its inverse
    #If the inverse is inf, set it to 0
    row_sum = np.array(mat.sum(1))
    rs_inv = np.power(row_sum, -1).flatten()
    rs_inv[np.isinf(rs_inv)] = 0
    r_inv_diag = sp.diags(rs_inv)
    
    mat = r_inv_diag.dot(mat)

    return mat




