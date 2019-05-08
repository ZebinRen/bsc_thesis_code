import tensorflow as tf
import numpy as np


def sparse_dropout(x, keep_prob, noise_shape):
    '''
    From kipf, GCN
    Do dropout for sparse tensors
    '''
    
    random_tensor = keep_prob

    #Add a random noise
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype = tf.bool)

    #Do the dropout
    pre_out = tf.sparse_retain(x, dropout_mask)
    out = pre_out * (1./keep_prob)

    return out


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

        

