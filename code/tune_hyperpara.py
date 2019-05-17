'''
This code will tune the hyperparameters
'''
import os

import tensorflow as tf
import pickle as pkl

from utils import *
from hyperpara_optim import *

from model.gcn import GCN
from model.mlp import MLP
from model.firstcheb import FirstCheb
from model.gat import GAT
from model.dcnn import DCNN
from model.spectralcnn import SpectralCNN
from model.chebnet import ChebNet
from model.graphsage import GraphSage

#Some parameters for tuning hyperparameters
random_times = 1
evaluate_times = 1

train_size = 230
val_size = 500

#mlp gcn firstcheb dcnn
model_list = [DCNN]
model_name_list = ['dcnn']

dataset_path = './data'
dataset_name = 'citeseer'
index_list = ['0', '1']

save_path = './hyperparameter'


#Additional parameters
dcnn_addi_parameter = {'hops': 3}

for i in range(len(model_list)):
    #Get model and model name
    model = model_list[i]
    model_name = model_name_list[i]

    #Get a list of data
    data_feed_train = []
    addi_parameter = None

    #Add addiparameters
    for index in index_list:
    	data, addi_parameter = create_input(model_name, dataset_path, dataset_name, index, train_size, val_size, None)
    	data = create_train_feed(data)
    	data_feed_train.append(data)

    if model_name == 'dcnn':
    	addi_parameter.update(dcnn_addi_parameter)

    #Random search and save parameters
    rand_set, rand_accu = random_search(model, data_feed_train, search_parameter, fixed_parameter, 
    	addi_parameter, random_times = random_times, evaluate_times = evaluate_times)

    random_search_file = open(os.path.join(save_path, model_name+'_'+'rand'), 'wb')
    pkl.dump((rand_set, rand_accu), random_search_file)
    random_search_file.close()

    '''
    #Dense search and save parameters
    dense_set, dense_accu = desne_search(model, rand_set, rand_accu, data_feed_train, search_parameter, 
    	fixed_parameter, addi_parameter)

    dense_search_file = open(os.path.join(save_path, model_name+'_'+'dense'), 'wb')
    pkl.dump((dense_set, dense_accu), dense_search_file)
    dense_search_file.close()
    '''
    

    
    







