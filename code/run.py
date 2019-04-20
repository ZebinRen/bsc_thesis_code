'''
This script is used to debug te model
Write the code in a new file to test the models
'''

import tensorflow as tf
from load_data import create_input
from utils import *
from model.gcn import GCN

learning_rate = 0.001
epochs = 20
weight_decay = 0 
early_stopping = 10
activation_func = tf.nn.relu
dropout_prob = None
bias = None

directed, undirected, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,\
info = create_input('./data', 'citeseer', 0, 500, 500, None)

#preprocess the adjancy matrix for GCN
sys_norm_directed = symmetric_normalized_laplacian(directed)
sys_norm_undirected = symmetric_normalized_laplacian(undirected)

#preprocess features
norm_features = row_normalized(features)

#information
nodes = directed.shape[0]
input_dim = features.shape[1]
output_dim = y_train.shape[1]

#Create model
model = GCN(
    hidden_num = 1, hidden_dim = [8],
    input_dim = input_dim, output_dim = output_dim,
    node_num = nodes, cate_num = output_dim,
    learning_rate = learning_rate, epochs = epochs,
    weight_decay = weight_decay, early_stopping = early_stopping,
    activation_func = activation_func,
    dropout_prob = dropout_prob,
    bias = bias,
    name='GCN'
)

sess = tf.Session()

model.train(sess, directed, features, y_train, train_mask)


