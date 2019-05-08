'''
This code will tune the hyperparameters
'''
import os

import tensorflow as tf
import pickle as pkl

from utils import *
from hyperpara_optim import *

from model.mlp import MLP
from model.gcn import GCN

#Some parameters for tuning hyperparameters
random_times = 50
evaluate_times = 3

model_list = [GCN, MLP]
model_name_list = ['gcn', 'mlp']

dataset_path = './data'
dataset_name = 'citeseer'

save_path = './hyperparameter'

for i in range(len(model_list)):
    #Get model and model name
    model = model_list[i]
    model_name = model_name_list[i]

    #Get input
    data, addi_parameter = create_input(model_name, './data', 'citeseer', 1, 230, 500, None)
    
    #Create train data, only train, and val part is used
    data_feed_train = create_train_feed(data)
    
    #Random search
    rand_set, rand_accu = random_search(model, data_feed_train, search_parameter, fixed_parameter, 
    	addi_parameter, random_times = random_times, evaluate_times = evaluate_times)

    dense_set, dense_accu = desne_search(model, rand_set, rand_accu, data_feed_train, search_parameter, 
    	fixed_parameter, addi_parameter)

    random_search_file = open(os.path.join(save_path, model_name+'_'+'rand'), 'wb')
    dense_search_file = open(os.path.join(save_path, model_name+'_'+'dense'), 'wb')

    #save parameters
    pkl.dump((rand_set, rand_accu), random_search_file)
    pkl.dump((dense_set, dense_accu), dense_search_file)

    random_search_file.close()
    dense_search_file.close()







