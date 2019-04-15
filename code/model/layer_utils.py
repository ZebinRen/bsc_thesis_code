import tensorflow as tf

def dropout_sparse(x, keep_prob):
    '''
    Do dropout for sparse tensors

    '''


def graph_conv(X, A, weights, A_is_sparse = False):
    '''
    Graph convolution:
    X: Feature Matrix(Can be sparse)
    A: Adjancy matrix, symmertic normalized Laplacian(Can be sparse)
    weights: weights(Can't be sparse)
    is_sparse: If the adjancy matrix is sparse

    Return: AXW

    '''
    XW = tf.matmul(X, W)
    
    if a_is_sparse:
        AXW = tf.sparse_tensor_dense_matmul(A, XW)
    else:
	AXW = tf.matmul(A, XW)
    
    return AXW

        

