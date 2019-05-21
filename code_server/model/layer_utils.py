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

def mask_by_adj(mat, adj):
    '''
    Mask a matrix by the adjancy matrix 
    The adj is a sparse matrix
    '''
    adj = tf.sparse_to_dense(adj.indices, adj.dense_shape, adj.values)
    mask = tf.cast(adj, tf.bool)
    mask = tf.cast(mask, tf.float32)
    #Element wise mul

    return tf.math.multiply(mat, mask)

def create_power_series(a, degree, sparse = False):
    '''
    compute the power series if A
    Used in dcnn
    mat is the matrix 
    degree is the degree of powers
    '''

    if sparse:
        #If the matrix is sparse matrix, convert it to dense matrix
        #exit()
        a = tf.sparse_to_dense(a.indices, a.dense_shape, a.values, validate_indices=False)

    pow_series = []
    pow_series.append(a)

    for i in range(degree - 1):
        pow_series.append(tf.matmul(pow_series[-1], a))

    return tf.stack(pow_series, 1)










        

