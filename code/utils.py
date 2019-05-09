import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from load_data import create_raw_input


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

    return [indexs, data, shape]

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

def create_train_feed(dataset, directed = False):
    '''
    create the parameters for train method
    '''
    if directed:
        adj = dataset['directed']
    else:
        adj = dataset['undirected']

    features = dataset['features']
    y_train = dataset['train_label']
    y_val = dataset['val_label']

    train_mask = dataset['train_mask']
    val_mask = dataset['val_mask']

    dataset = {
        'adj': adj, 'features': features, 
        'train_label': y_train, 
        'val_label': y_val, 
        'train_mask': train_mask, 
        'val_mask': val_mask
    }

    return dataset

def pre_GCN(directed, undirected):
    '''
    Preprocess adjancy matrix for GCN
    '''
    sys_norm_directed = symmetric_normalized_laplacian(directed)
    sys_norm_undirected = symmetric_normalized_laplacian(undirected)

    return sys_norm_directed, sys_norm_undirected

def pre_DCNN(directed, undirected):
    '''
    Preprocess for dcnn
    Build degree normalized transition matrix
    '''
    directed = row_normalized(directed)
    undirected = row_normalized(undirected)

    return directed, undirected





def create_input(model_name, path, dataset_name, index, train_num, val_num, test_num = None):
    '''
    This will create the input that can be directly feed to the neural network
    '''
    directed, undirected, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,\
    info = create_raw_input('./data', 'citeseer', 1, 230, 500, None)

    #preprocess features
    norm_features = row_normalized(features)

    #information
    node_num = directed.shape[0]
    input_dim = features.shape[1]
    output_dim = y_train.shape[1]

    #Preprocess adjancy for different models
    if 'gcn' == model_name:
        directed, undirected = pre_GCN(directed, undirected)
    elif 'dcnn' == model_name:
        directed, undirected = pre_DCNN(directed, undirected)
    elif 'gat' == model_name:
        #directe, sys_norm_undirected
        pass
    else:
        raise 'There is no model named: ' + model_name

    
    #Change scipy sparse matrix to the format that can be directly used by
    #the model
    directed = create_load_sparse(directed)
    undirected = create_load_sparse(undirected)
    features = create_load_sparse(features)

    dataset = {
        'directed': directed,
        'undirected': undirected,
        'features': features,
        'train_label': y_train,
        'val_label': y_val,
        'test_label': y_test,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }

    info = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'node_num': node_num,
        'cate_num': output_dim
    }

    return dataset, info




