import tensorflow as tf
import numpy as np

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
    XW = tf.matmul(X, weights)
    
    if a_is_sparse:
        AXW = tf.sparse_tensor_dense_matmul(A, XW)
    else:
        AXW = tf.matmul(A, XW)
    
    return AXW


def glort_init(shape, name=None):
    '''
    Glort init
    '''
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    init = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)

    return tf.Variable(init, name=name)

        

