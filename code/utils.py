import tensorflow as tf
import scipy.sparse as sp
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
    rowsum = np.array(adj.sum(1))

    #Compute D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    #If it is inf(The inverse of 0, then set it to 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    return normalized
    


def row_normalized(mat):
    '''
    row normalized feature
    '''
    #Compute row sum and its inverse
    #If the inverse is inf, set it to 0
    row_sum = np.array(mat.sum(1))
    rs_inv = np.power(row_sum, -1).flatten
    rs_inv[np.isinf(rs_inv)] = 0
    r_inv_diag = sp.diags(rs_inv)
    
    mat = r_inv_diag.dot(mat)

    return mat




