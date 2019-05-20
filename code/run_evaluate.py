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

import os


model_list = [MLP]
model_name_list = ['mlp']

dataset_name = 'cora'
dataset_numbers = 2
parameter_appendix = 'rand'

dataset_path = './data/evaluate'
parameter_path = './hyperparameter'
result_path = 'result_1'

evaluate_times = 2
train_size = 230
val_size = 500 


for model, model_name in zip(model_list, model_name_list): 
    train_info_list, acc_list, time_list = evaluate_model(model, 
            model_name, dataset_path, dataset_name, dataset_numbers, parameter_path, 
            parameter_appendix, result_path, evaluate_times, train_size, val_size)
