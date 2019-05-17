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

'''
This script will run the evaluation
The following parameters should be provided
A list of paths to the dataset
Path where the result are saved
'''

dataset_path_list = 
dataset_name_list = 

