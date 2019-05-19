import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from load_data import create_raw_input
from scipy.sparse.linalg.eigen.arpack import eigsh


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

def create_cheb_series(adjancy, poly_order, self_loop=True):
    '''
    Used to build ChebNet input
    compute cheb_series
    Used in ChebNet
    input is sparse matrix
    this should be processed before it is feeded into the model
    using numpy
    self_loop: if the adjancy matrix has a selfloop
    '''

    ##Normalize adjancy
    ##L= D - W
    ##D is row_sum
    W = symmetric_normalized_laplacian(adjancy)
    W = sp.coo_matrix(adjancy)
    D = np.array(W.sum(1))
    D = D.flatten()
    shape = W.shape

    L = sp.diags(D) - W

    if self_loop:
        L = L + sp.eye(shape[0])

    #Get the largest eigenvalue
    l_ev = eigsh(L + L.T,1,which='LA')[0]
    l_ev = l_ev[0]
    print(l_ev)
    #exit()
    #l_ev = 1

    L_hat = (2*L)/l_ev - sp.eye(shape[1])

    cheb_series = []

    cheb_series.append(sp.eye(shape[0]))
    cheb_series.append(L_hat)

    for i in range(2, poly_order):
        L_cp = sp.csr_matrix(L, copy=True)
        res = 2*L_cp.dot(cheb_series[-1]) - cheb_series[-2]
        cheb_series.append(res)

    undirected = [create_load_sparse(item) for item in cheb_series]

    return undirected

def create_mean_pool_adj_info(adjancy):
    '''
    Create the neighborhood informaion for GraphSage
    Used by mean pool
    '''
    adjancy = sp.coo_matrix(adjancy)

    row = adjancy.row
    col = adjancy.col

    return row, col

def create_neighbor_matrix(adjancy, num_nodes, maxdegree):
    '''
    Create the neighborhood matrix
    '''
    adjancy = sp.coo_matrix(adjancy)

    neigh = np.zeros((num_nodes, maxdegree), dtype=np.int32)
    loc = np.zeros((num_nodes), dtype=np.int32)

    #get row and column index
    row = adjancy.row
    col = adjancy.col

    for index in zip(row, col):
        node = index[0]
        value = index[1]
        locate = loc[node]

        #update neighborhood information
        neigh[node][locate] = value 

        #update location
        loc[node] = locate + 1

    return neigh





def create_input(model_name, path, dataset_name, index, train_num, val_num, test_num = None):
    '''
    This will create the input that can be directly feed to the neural network
    '''
    directed, undirected, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,\
    info = create_raw_input(path, dataset_name, index, 230, 500, None)

    #preprocess features
    norm_features = row_normalized(features)

    #information
    node_num = directed.shape[0]
    input_dim = features.shape[1]
    output_dim = y_train.shape[1]

    #return value
    dataset = {}
    info = {}

    #create degrees
    binary_value = undirected.astype(np.bool)
    binary_value = binary_value.astype(np.int32)
    degrees = np.array(binary_value.sum(1))
    maxdegree = np.max(degrees)

    #create neigh_info, used by graphsage max pool
    neigh_info = create_neighbor_matrix(undirected, node_num, maxdegree)
    row, col = create_mean_pool_adj_info(undirected)

    #Preprocess adjancy for different models
    if 'gcn' == model_name or 'firstcheb' == model_name:
        directed, undirected = pre_GCN(directed, undirected)
    elif 'dcnn' == model_name:
        directed, undirected = pre_DCNN(directed, undirected)
    elif 'spectralcnn' == model_name:
        #Convert to dense matrix
        #only the undirected matrix is computed
        #Since the directed adjancy is not used in any model
        dense_undirected = sp.csr_matrix.todense(undirected)

        #compute eigenvalue decompsition
        undirected_evalues, undirected_evectors = np.linalg.eigh(dense_undirected)
        undirected = undirected_evectors
    elif 'chebnet' == model_name:
        pass
    elif 'gat' == model_name:
        dataset['row'] = row
        dataset['col'] = col
        indices = zip(row, col)
        indices = [ind for ind in indices]
        dataset[indices] = indices
    elif 'graphsage' == model_name:
        dataset['degrees'] = degrees
    elif 'graphsage_maxpool' == model_name:
        info['max_degree'] = maxdegree
        dataset['degrees'] = degrees
        dataset['neigh_info'] = neigh_info
    elif 'graphsage_meanpool' == model_name:
        dataset['degrees'] = degrees
        dataset['row'] = row
        dataset['col'] = col
    elif 'mlp' == model_name:
        pass
    else:
        raise 'There is no model named: ' + model_name

    
    #Change scipy sparse matrix to the format that can be directly used by
    #the model
    if 'spectralcnn' == model_name:
        #Adjancy matrix is not used in these models
        #The eigenvector is used
        #directed = None
        #print(undirected_evectors.shape)
        #exit()
        #undirected = [undirected_evalues, undirected_evectors]
        pass
    elif 'chebnet' == model_name:
        pass
    else:
        directed = create_load_sparse(directed)
        undirected = create_load_sparse(undirected)
    
    features = create_load_sparse(features)


    dataset.update({
        'directed': directed,
        'undirected': undirected,
        'features': features,
        'train_label': y_train,
        'val_label': y_val,
        'test_label': y_test,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    })

    info.update({
        'input_dim': input_dim,
        'output_dim': output_dim,
        'node_num': node_num,
        'cate_num': output_dim
    })

    return dataset, info




