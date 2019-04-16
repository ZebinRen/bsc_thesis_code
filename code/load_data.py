import os
import pickle

import numpy as np

def create_mask(index, length):
    '''
    Create mask:
    index is the indexs that should be 1, index is a list
    length is the length of the mask
    '''
    mask = np.zeros(length)
    mask[index] = 1
    mask = np.array(mask, dtype=np.bool)

    return mask

def load_data(path, dataset_name, index):
    '''
    Load processed data from the file
    The splition is the sequence of the nodes
    '''
    name_list_origin = ['directed', 'undirected', 'features', 'cate']

    #Create the name list of the data file
    name_list = []
    for name in name_list_origin:
        name_list.append(dataset_name + '_' + name + '_' + str(index))
    
    #Read file
    data_list = []
    for name in name_list:
        file_cur = open(os.path.join(path, name), 'rb')
        data_list.append(pickle.load(file_cur))
        file_cur.close()
    
    #Read info file, each dataset only has one info file
    file_cur = open(os.path.join(path, dataset_name + '_info'), 'rb')
    data_list.append(pickle.load(file_cur))
    file_cur.close()

    #Create data dictionary
    data_dict = {}
    name_list_origin.append('info')
    for i in range(len(data_list)):
        data_dict[name_list_origin[i]] = data_list[i]


    return data_dict


def create_input(path, dataset_name, index, train_num, val_num, test_num = None):
    '''
    Create the data that can be directly used by the neural network
    NOTE: The data is not normalized
    The train() method in the model class will do it
    If test_num is None, then the sizeof test_num is calcualte automatically
    '''
    #Read data
    data_dict = load_data(path, dataset_name, index)

    #Get the number of nodes and the number of cates
    #info has two keys: node_num, cate_num
    info = data_dict['info']
    node_num = info['node_num']
    cate_num = info['cate_num']

    #If size is not provided
    if None == test_num:
        test_num = node_num - train_num - val_num
        if test_num <= 0:
            raise "test_num should be a positive value"
        else:
            print("test_num is calculated automatically")
            print("test_num =", test_num)
    
    #Check if the sum of train_num, val_num, test_num is node_num
    if node_num == train_num + val_num + test_num:
        pass
    else:
        raise "Incompatible arguements: The sum of train, val, test is not nodes_num"
    

    #Create mask
    index_train = list(range(train_num))
    index_val = list(range(train_num, train_num + val_num))
    index_test = list(range(train_num + val_num, node_num))

    train_mask = create_mask(index_train, node_num)
    val_mask = create_mask(index_val, node_num)
    test_mask = create_mask(index_test, node_num)

    #Create train/val/test labels
    y_train = np.zeros((node_num, cate_num))
    y_val = np.zeros((node_num, cate_num))
    y_test = np.zeros((node_num, cate_num))

    cate = data_dict['cate']
    y_train[train_mask, :] = cate[train_mask, :]
    y_val[val_mask, :] = cate[val_mask,:]
    y_test[test_mask, :] = cate[test_mask,: ]

    return data_dict['directed'], data_dict['undirected'], data_dict['features'], y_train, y_val,\
            y_test, train_mask, val_mask, test_mask, data_dict['info']


#run test
directed, undirected, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,\
info = create_input('./data', 'citeseer', 0, 500, 500, None)

