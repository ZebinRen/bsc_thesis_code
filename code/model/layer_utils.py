import tensorflow as tf

def graph_conv(X, A, weights, x_is_sparse = False, a_is_sparse = False):
    '''
    Graph convolution:
    X: Feature Matrix(Can be sparse)
    A: Adjancy matrix, renormalized Laplacian(Can be sparse)
    weights: weights(Can't be sparse)
    is_sparse: If the input matrix is sparse

    Return: AXW

    '''
    if x_is_sparse:
        XW = tf.sparse_tensor_dense_matmul(X, W)
    else:
        XW = tf.matmul(X, W)
    
    if a_is_sparse:
        AXW = tf.sparse_tensor_dense_matmul(A, XW)
    else:
	AXW = tf.matmul(A, XW)
    
    return AXW

        

