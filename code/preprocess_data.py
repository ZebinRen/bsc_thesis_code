'''
In this file, the processed orgin data will be read
Then it will create 10 groups of data which can be directly used
for training, validating and testing
'''

import numpy as np
import pickle
import scipy.sparse as sp
import os

from random import shuffle

'''
There are four files for each dataset
datasetname_directed: Adjancy matrix, directed(CSR format)
datasetname_undirected: Adjancy matrix, undirected(CSR format)
datasetname_attribute: Feature matrix(CSR format)
datasetname_cate: Cate, (dict(node_number: cate))
datasetname_info: Stores the information of data, (dict)
'''
def read_origin_data(read_path, dataset):
    name_list = ['_directed', '_undirected', '_attribute', '_cate', '_info']
    data = []

    for name in name_list:
        read_file = open(os.path.join(read_path, dataset+name), 'rb')
        data.append(pickle.load(read_file))
        read_file.close()
    
    return data[0] , data[1], data[2], data[3], data[4]

def save_data(path, name_list, data_list):
    '''
    Save a list of data
    name_list: names
    data_list: objects to be saved
    '''
    if len(name_list) != len(data_list):
        raise "Name_list and data_list doesn't match"

    #open file   
    file_list = []

    for i in range(len(name_list)):
        file_list.append(open(os.path.join(path, name_list[i]), 'wb'))
    
    #Save data
    for i in range(len(file_list)):
        pickle.dump(data_list[i], file_list[i])
    
    #Close file
    for file_object in file_list:
        file_object.close()
    

def sparse_to_tuple(sparse_mat):
    '''
    Convert sparse matrix to tuple representation
    Return value:
    [list_of_indexs, list_of_values, shape]
    '''
    if not sp.isspmatrix_coo(sparse_mat):
        sparse_mat = sparse_mat.tocoo()
    indexs = np.vstack((sparse_mat.row, sparse_mat.col)).transpose()
    data = sparse_mat.data
    shape = sparse_mat.shape

    return [indexs, data, shape]

def tuple_to_sparse(tuple_mat):
    '''
    Convert tuple represented matrix to sparse matrix
    Return value: CSR(Compressed row matrix) 
    '''
    indexs = tuple_mat[0]
    data = tuple_mat[1]
    shape = tuple_mat[2]

    row, col = indexs.transpose()

    return sp.csr_matrix((data, (row, col)), shape = shape)

    


def shuffle_dataset(directed, undirected, features, cate, nodes_num, cate_num):
    '''
    Shuffle, used to shuffle the dataset
    Create multiple train/validation/test split
    Arguments:
    directed, undirected, features: sparse
    cate: dict
    Return:

    '''

    #Create shuffle list
    shuffle_list = list(range(nodes_num))
    shuffle(shuffle_list)

    #Change adjancy matrix to tuple representation
    directed = sparse_to_tuple(directed)
    undirected = sparse_to_tuple(undirected)

    #Shuffle adjancy matrix using shuffle list
    #And convert them to sparse matrix
    directed_index_list = directed[0]
    undirected_index_list = undirected[0]
    
    for index in directed_index_list:
        index[0] = shuffle_list[index[0]]
        index[1] = shuffle_list[index[1]]

    for index in undirected_index_list:
        index[0] = shuffle_list[index[0]]
        index[1] = shuffle_list[index[1]]

    directed = tuple_to_sparse(directed)
    undirected = tuple_to_sparse(undirected)    
    
    #Shuffle features
    #And convert it to sparse matrix
    features = sparse_to_tuple(features)
    features_index_list = features[0]

    for index in features_index_list:
        index[0] = shuffle_list[index[0]]

    features = tuple_to_sparse(features)

    #Shuffle cate
    #The new cate is one-hot representation
    new_cate = np.zeros(shape=(nodes_num, cate_num), dtype=np.int32)

    #print(cate)
    for key in cate:
        new_cate[shuffle_list[key]][cate[key]] = 1
    
    cate = new_cate

    return directed, undirected, features, cate

    
def create_feed_dataset(dataset_read, dataset_save, dataset_name, instance_num):
    '''
    Create dataset that can be directly feeded to the neural network
    Multiple instances will be created
    '''
    directed, undirected, features, cate, info = read_origin_data(dataset_read, dataset_name)
    name_list_origin = ['directed', 'undirected', 'features', 'cate']

    #Create namelist used to save the objects
    #dataset's named is added as prefix
    name_list_save = []
    for name in name_list_origin:
        name_list_save.append(dataset_name + '_' + name)


    #Read info
    node_num = info['node_num']
    cate_num = info['cate_num']

    #Suffle data and save them 
    for i in range(instance_num):
        new_dir, new_undir, new_feat, new_cate = shuffle_dataset(directed, undirected, 
                                                        features, cate, node_num, cate_num)
        #Get the name_list used to save the current data
        #The index of the current set of data is added as postfix
        name_list_save_current = []
        for name in name_list_save:
            name_list_save_current.append(name + '_' + str(i))
        
        data_list = [new_dir, new_undir, new_feat, new_cate]
        
        #Save
        save_data(dataset_save, name_list_save_current, data_list)
    
    #Save info
    save_data(dataset_save, [dataset_name + '_info'], [info])


#This will create the dataset used for neural networks
def create_dataset(read_path, save_path, instance_num):
    #Read dataset: citeseer, cora, pubmed
    create_feed_dataset(read_path, save_path, 'citeseer', instance_num)
    create_feed_dataset(read_path, save_path, 'cora', instance_num)
    create_feed_dataset(read_path, save_path, 'pubmed', instance_num)


if __name__ == '__main__':
    #Create for hyperparameter
    create_dataset('../processed_data', './data/tune_hyper', 3)

    #Create for evaluation
    create_dataset('../processed_data', './data/evaluate', 10)

