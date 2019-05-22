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
import os

from hyperpara_optim import create_train_feed
from hyperpara_optim import create_test_feed

'''
This script will run the evaluation
The following parameters should be provided
A list of paths to the dataset
Path where the result are saved
'''

def evaluate(model, model_name, parameters, train_dataset_list, test_dataset_list, evaluate_times = 2):
    '''
    This function will evaluate a model on a sinlge dataset
    Multiple instances for the same dataset can be used
    model: The model to be tested
    model_name: Name of the model. used in saving the result
    parameter: Hyperparameters
    dataset_list: A list of dataset
    evaluate_times: How many times for each dataset
    '''
    #Test informations
    train_info_list = []
    accu_list = []
    time_list = []

    for train_dataset, test_dataset in zip(train_dataset_list, test_dataset_list):
        cur_train_info_list = []
        cur_accu_list = []
        cur_time_list = []
        for each_eva in range(evaluate_times):
            sess = tf.Session()
            test_model = model(**parameters)
            train_info = test_model.train(sess, **train_dataset)
            test_accu, test_time = test_model.test(sess, **test_dataset)

            #Append current train info
            cur_train_info_list.append(train_info)
            cur_accu_list.append(test_accu)
            cur_time_list.append(test_time)

        #Append to global
        train_info_list.append(cur_train_info_list)
        accu_list.append(cur_accu_list)
        time_list.append(cur_time_list)

    return train_info_list, accu_list, time_list

def load_evaluate_dataset(model_name, dataset_path, dataset_name, index_list, train_size, val_size):
    '''
    load dataset for evaluation
    path: dataset path
    name: dataset name
    index: a list of indexs
    train_size: number of nodes used in train
    val_size: number of nodes used in validation
    return value:
    train_dataset_list: the dataset list which can be directly feed to train
    test_dataset_list: the dataset list which can be directly feed to test
    '''
    train_dataset_list = []
    test_dataset_list = []
    addi_parameter = None


    for cur_index in index_list:
        data, addi_parameter = create_input(model_name, dataset_path, dataset_name, cur_index, train_size, val_size, None)
        data_train = create_train_feed(data, model_name)
        data_val = create_test_feed(data, model_name)
        train_dataset_list.append(data_train)
        test_dataset_list.append(data_val)

    return train_dataset_list, test_dataset_list, addi_parameter


def evaluate_model(model, model_name, dataset_path, dataset_name, dataset_numbers, 
    parameter_path, parameter_appendix, result_path, evaluate_times,
    train_size, val_size, save = False):
    '''
    model: The model class
    model_name: name of the model,
    dataset_path: path to dataset
    dataset_numbers: numbers of datasets used in the test
    paraemter_path: path of parameter
    parameter_appendix: appendix in parameter file name
    evaluate_time: how many times of evalation for each model
    train_size: The number of nodes used in train
    val_size: The number nodes used in validation
    '''

    #Some preprocess
    dataset_index_list = [str(i) for i in range(dataset_numbers)]

    #Read parameters
    para_file = open(os.path.join(parameter_path, model_name + '_' + parameter_appendix), 'rb')
    parameters = pkl.load(para_file)[0]
    para_file.close()

    #add
    if model_name == 'dcnn' and dataset_name == 'pubmed':
        parameters.update({'hops': 2})

    #Read train data
    train_dataset_list, test_dataset_list, addi_parameter = load_evaluate_dataset(model_name, 
        dataset_path, dataset_name, dataset_index_list, train_size, val_size)
    parameters.update(addi_parameter)


    #evaluate the data
    train_info_list, accu_list, time_list = evaluate(model, model_name, 
        parameters, train_dataset_list, test_dataset_list, evaluate_times)

    #save the result to file
    if save:
        result_file_name = model_name + '_' + 'result'
        result_file = open(os.path.join(result_path, result_file_name), 'wb')
        pkl.dump((train_info_list, accu_list, time_list), result_file)

        result_file.close()

    return train_info_list, accu_list, time_list


