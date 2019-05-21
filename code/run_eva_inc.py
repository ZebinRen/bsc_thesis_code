'''
This file will test the affective of increasing train data
The data is based on citeseer
'''
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
from model.graphsage_maxpool import GraphSageMaxPool
from model.graphsage_meanpool import GraphSageMeanPool
from hyperpara_optim import *
import scipy.sparse as sp
import numpy as np
import pickle as pkl
from process_data import *
from draw import *

import os

model_list = [FirstCheb]
model_name_list = ['firstcheb']

train_size_list = [20, 40, 60, 100, 140, 180, 220, 260, 300, 400, 500, 600, 800, 1000, 1200, 1400]

dataset_name = 'citeseer'
dataset_numbers = 10
parameter_appendix = 'rand'

dataset_path = './data/evaluate'
parameter_path = './hyperparameter'
result_path = './direct_inc_output'
processed_result_path = './processed_inc_output'

evaluate_times = 1
val_size = 500

for model, model_name in zip(model_list, model_name_list):
    global_acc_list = []
    for train_size in train_size_list:
        train_info_list, acc_list, time_list = evaluate_model(model, 
            model_name, dataset_path, dataset_name, dataset_numbers, parameter_path, 
            parameter_appendix, result_path, evaluate_times, train_size, val_size)

        #save to file
        save_path = result_path + '_' + model_name
        file_name = model_name + parameter_appendix + '_' + str(train_size) 
        #make directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_file = open(os.path.join(save_path, file_name), 'wb')
        pkl.dump((train_info_list, acc_list, time_list), save_file)
        save_file.close()

        #process output data
        train_info, acc, time = process_output(train_info_list, acc_list, time_list)

        #save processed data
        save_path = processed_result_path + '_' + model_name
        file_name = model_name + parameter_appendix + '_' + str(train_size)


        #make directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_file = open(os.path.join(save_path, file_name), 'wb')
        pkl.dump((train_info, acc, time), save_file)
        save_file.close()

        #Add to global list
        global_acc_list.append(acc)

    #Save the global_acc_list
    save_path = processed_result_path + '_global'
    file_name = dataset_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    save_file = open(os.path.join(save_path, file_name), 'wb')
    pkl.dump(global_acc_list, save_file)
    save_file.close()

    file_name = dataset_name + '.txt'
    save_file = open(os.path.join(save_path, file_name), 'w')
    save_file.writelines(['size, accuracy, accuracy_std'])

    for cur_size, cur_acc in zip(train_info_list, global_acc_list):
        cur_line = str(cur_size)
        cur_line = cur_line + ',' + str(cur_acc[0]) + ',' + str(cur_acc[1])
        save_file.writelines([cur_line])

    save_file.close()






