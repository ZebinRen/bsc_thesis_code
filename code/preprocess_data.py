'''
In this file, the processed orgin data will be read
Then it will create 10 groups of data which can be directly used
for training, validating and testing
'''

import numpy as np
import pickle
import scipy.sparse as sp
import os

def read_origin_data(read_path, dataset):
    name_list = ['_directed', '_undirected', '_attribute', '_cate']
    data = []

    for name in name_list:
        read_file = open(os.path.join(read_path, dataset+name), 'rb')
        data.append(pickle.load(read_file))
    
    return data[0] , data[1], data[2], data[3]