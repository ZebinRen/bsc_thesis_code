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
from model.graphsage_meanpool import GraphSageMeanPool
from model.graphsage_maxpool import GraphSageMaxPool

#Some parameters for tuning hyperparameters
random_times = 30
evaluate_times = 3

train_size = 230
val_size = 500

'''
#mlp gcn firstcheb dcnn graphsage graph_max_pool graph_mean_pool spectralcnn
model_list = [MLP, GCN, FirstCheb, DCNN, GraphSage, GraphSageMaxPool, GraphSageMeanPool, GAT, SpectralCNN]
model_name_list = ['mlp', 'gcn', 'firstcheb', 'dcnn', 'graphsage', 'graphsage_maxpool', 
				 'graphsage_meanpool', 'gat', 'spectralcnn']
'''
model_list = [GCN]
model_name_list = ['gcn']

dataset_path = './data/tune_hyper'
dataset_name = 'citeseer'
index_list = ['0', '1', '2']
#index_list = ['0']

save_path = './hyperparameter'


#Additional parameters
dcnn_addi_parameter = {'hops': 3}


graphsage_meanpool_addi_parameter = {'transform_size': [32, 32]}
graphsage_maxpool_addi_parameter = {'transform_size': [32, 32]}

gat_addi_parameter = {'attention_head': 3}

for i in range(len(model_list)):
    #Get model and model name
    model = model_list[i]
    model_name = model_name_list[i]

    #Get a list of data
    data_feed_train = []
    data_feed_val = []
    addi_parameter = None

    #Add addiparameters
    for index in index_list:
    	data, addi_parameter = create_input(model_name, dataset_path, dataset_name, index, train_size, val_size, None)
    	data_train = create_train_feed(data, model_name)
    	data_val = create_test_feed(data, model_name)
    	data_feed_train.append(data_train)
    	data_feed_val.append(data_val)

    if model_name == 'dcnn':
    	addi_parameter.update(dcnn_addi_parameter)
    elif model_name == 'graphsage_meanpool':
    	addi_parameter.update(graphsage_meanpool_addi_parameter)
    elif model_name == 'graphsage_maxpool':
    	addi_parameter.update(graphsage_maxpool_addi_parameter)
    elif model_name == 'gat':
    	addi_parameter.update(gat_addi_parameter)

    #Random search and save parameters
    rand_set, rand_accu = random_search(model, data_feed_train, data_feed_val, search_parameter, fixed_parameter, 
    	addi_parameter, random_times = random_times, evaluate_times = evaluate_times)

    random_search_file = open(os.path.join(save_path, model_name+'_'+'rand'), 'wb')
    pkl.dump((rand_set, rand_accu), random_search_file)
    random_search_file.close()

    
    #Dense search and save parameters
    dense_set, dense_accu = desne_search(model, rand_set, rand_accu, data_feed_train, data_feed_val, search_parameter, 
    	{}, {}, evaluate_times)

    dense_search_file = open(os.path.join(save_path, model_name+'_'+'dense'), 'wb')
    pkl.dump((dense_set, dense_accu), dense_search_file)
    dense_search_file.close()
    
    

    
    







