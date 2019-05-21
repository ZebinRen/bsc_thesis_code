from evaluate import *
import tensorflow as tf
from utils import *
from model.gcn import GCN
from model.mlp import MLP
from model.firstcheb import FirstCheb
from model.gat import GAT
from model.dcnn import DCNN
from model.spectralcnn import SpectralCNN
from model.chebnet import ChebNet
from model.graphsage import GraphSage
from hyperpara_optim import *
import scipy.sparse as sp
import numpy as np
import pickle as pkl
from process_data import *
from draw import *

import os

'''
This file will run the test script
Three kinds of file are saved
result_path/dataset_name: the original data 
processed_result_path/dataset_name/: processed data
'''

#Model done
#MLP
model_list = [MLP]
model_name_list = ['mlp']

dataset_name_list = ['citeseer', 'cora', 'pubmed']
dataset_numbers = 10
parameter_appendix_list = ['rand']

dataset_path = './data/evaluate'
parameter_path = './hyperparameter'
result_path = './direct_output'
processed_result_path = './processed_output'

evaluate_times = 2
train_size = 230
val_size = 500 


for model, model_name in zip(model_list, model_name_list):
    for dataset_name in dataset_name_list:
        for parameter_appendix in parameter_appendix_list:
            train_info_list, acc_list, time_list = evaluate_model(model, 
                model_name, dataset_path, dataset_name, dataset_numbers, parameter_path, 
                parameter_appendix, result_path, evaluate_times, train_size, val_size)

            #save to file
            save_path = os.path.join(result_path, dataset_name)
            file_name = model_name + parameter_appendix
            #make directory
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_file = open(os.path.join(save_path, file_name), 'wb')
            pkl.dump((train_info_list, acc_list, time_list), save_file)
            save_file.close()


            #process output data
            train_info, acc, time = process_output(train_info_list, acc_list, time_list)

            #save processed data
            save_path = os.path.join(processed_result_path, dataset_name)
            file_name = model_name + parameter_appendix


            #make directory
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_file = open(os.path.join(save_path, file_name), 'wb')
            pkl.dump((train_info, acc, time), save_file)
            save_file.close()

            #save train image
            save_path = os.path.join(processed_result_path, dataset_name)
            plot_train(train_info['train_loss'], train_info['train_acc'], 
                train_info['val_loss'], train_info['val_acc'],
                save_path, model_name, True)


