import tensorflow as tf
import numpy as np

'''
def sparse_dropout(x, keep_prob, dim):
    
    Do dropout for sparse tensors

    
    noise_length = x.values.shape[0]
    print(noise_length)

    dropout_mask = keep_prob + tf.random_uniform((noise_length))
    dropout_mask = tf.floor(dropout_mask)
    dropout_mask = tf.cast(dropout_mask, dtype=tf.bool)

    outputs = tf.sparse_retain(x, dropout_mask)
    outputs = outputs * (1/keep_prob)

    return outputs
'''

def graph_conv(X, A, weights, X_is_sparse = False):
    '''
    Graph convolution:
    X: Feature Matrix(Can be sparse)
    A: Adjancy matrix, symmertic normalized Laplacian(Can be sparse)
    weights: weights(Can't be sparse)
    is_sparse: If the adjancy matrix is sparse

    Return: AXW

    '''
    print(X.shape)
    print(weights.shape)
    if X_is_sparse:
        XW = tf.sparse_tensor_dense_matmul(X, weights)
    else: 
        XW = tf.matmul(X, weights)
    
    AXW = tf.sparse_tensor_dense_matmul(A, XW)

    
    return AXW


def glort_init(shape, name=None):
    '''
    Glort init
    '''
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    init = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)

    return tf.Variable(init, name=name)

        

