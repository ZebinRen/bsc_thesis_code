'''
This script is used to debug te model
Write the code in a new file to test the models
'''

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

#The model name used for test
model_name = 'gcn'
test_search_hpara = False

#Hyperparameters
learning_rate = 0.01 #0.01
epochs = 400
weight_decay = 0.0001 #5e-1
early_stopping = 30 #500
activation_func = tf.nn.relu
dropout_prob = 0.1 #0.5
bias = True
hidden_dim = 16
optimizer = tf.train.AdamOptimizer


#hyperparameter for ChebNet 
poly_order = 2

#train size
train_size = 10

#load data
data, addi_parameters = create_input(model_name, './data', 'citeseer', '0', 230, 500, None)
directed = data['directed']
undirected = data['undirected']
features = data['features']
y_train = data['train_label']
y_val = data['val_label']
y_test = data['test_label']
train_mask = data['train_mask']
val_mask = data['val_mask']
test_mask = data['test_mask']
num_featuers_nonzero = features[1].shape






'''
#Test random search
dataset = create_train_feed(data)

para_set, para_accu = random_search(GCN, dataset, search_parameter, fixed_parameter, addi_parameters, random_times = 10,
    evaluate_times = 3)

print(para_set)
print(para_accu)

sess = tf.Session()

model = GCN(**para_set, **fixed_parameter, **addi_parameters)
model.train(sess, undirected, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero)


accu = model.test(sess, undirected, features, y_test, test_mask)
print('test acucracy: ', accu)

para_set, para_accu = desne_search(GCN, para_set, para_accu, dataset, search_parameter, fixed_parameter, addi_parameters)

model = GCN(**para_set, **fixed_parameter, **addi_parameters)
model.train(sess, undirected, features, y_train, y_val, train_mask, val_mask)


accu = model.test(sess, undirected, features, y_test, test_mask)
print(para_set)
print(para_accu)
print('test acucracy: ', accu)


#Test random search finish
'''


#Test GCN
if model_name == 'gcn':
#Create model
    model = GCN(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        name='GCN'
    )

    sess = tf.Session()

    model.train(sess, undirected, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero)


    accu, time_used = model.test(sess, undirected, features, y_test, test_mask, num_featuers_nonzero)
    print('test acucracy: ', accu)

#Test GCN finish




#Test MLP
elif model_name == 'mlp':
    #Create model
    model = MLP(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        name='MLP'
    )

    sess = tf.Session()

    model.train(sess, directed, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero)


    accu, time_used = model.test(sess, directed, features, y_test, test_mask, num_featuers_nonzero)
    print('test acucracy: ', accu)

#Test MLP finish


#Test first cheb
elif model_name == 'firstcheb':
    #print('#####', num_featuers_nonzero)
    #exit()
    
    model = FirstCheb(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        name='FirstCheb'
    )

    sess = tf.Session()
    
    model.train(sess, directed, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero)


    accu, time_used = model.test(sess, directed, features, y_test, test_mask, num_featuers_nonzero)
    print('test acucracy: ', accu)
#Test first cheb finish



#Test GAT
elif model_name =='gat':
    row = data['row']
    col = data['col']
    indices = data['indices']

    model = GAT(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        attention_head = 3,
        name='GAT'
    )

    sess = tf.Session()

    model.train(sess, directed, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero, row, col, indices)


    accu, _ = model.test(sess, directed, features, y_test, test_mask, num_featuers_nonzero, row, col, indices)
    print('test acucracy: ', accu)

#TEST GAT FINISH


#TEST DCNN
elif model_name == 'dcnn':
    model = DCNN(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        name='DCNN',
        hops = 3
    )

    sess = tf.Session()

    model.train(sess, undirected, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero)


    accu, time_used = model.test(sess, undirected, features, y_test, test_mask, num_featuers_nonzero)
    print('test acucracy: ', accu)

#TEST DCNN FINISHED

#TEST SpectralCNN
elif model_name == 'spectralcnn':

    model = SpectralCNN(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        name='SpectralCNN',
    )

    sess = tf.Session()

    model.train(sess, undirected, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero)


    accu, time_used = model.test(sess, undirected, features, y_test, test_mask, num_featuers_nonzero)
    print('test acucracy: ', accu)

#TEST SpectralCNN finish


#Test chebnet
elif model_name == 'chebnet':
	#create chebnet input
    cheb_series = create_cheb_series(undirected, poly_order, self_loop=True)
    undirected = cheb_series

    model = ChebNet(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        name='SpectralCNN',
        poly_order = poly_order
    )

    sess = tf.Session()

    model.train(sess, undirected, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero)


    accu = model.test(sess, undirected, features, y_test, test_mask, num_featuers_nonzero)
    print('test acucracy: ', accu)



#Test chebnet finish

#Test graphsage mean
elif model_name == 'graphsage':
    degrees = data['degrees']

    model = GraphSage(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        name='GraphSage'
    )

    sess = tf.Session()

    model.train(sess, undirected, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero, degrees)


    accu, time_used = model.test(sess, undirected, features, y_test, test_mask, num_featuers_nonzero, degrees)
    print('test acucracy: ', accu)

#Test graphsage mean finish

#Test graphsage maxpool
elif model_name == 'graphsage_maxpool':
    degrees = data['degrees']
    neigh_info = data['neigh_info']

    model = GraphSageMaxPool(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        name='GraphSage',
        transform_size = [32, 24]
    )

    sess = tf.Session()

    model.train(sess, undirected, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero, degrees, neigh_info)


    accu, time_used = model.test(sess, undirected, features, y_test, test_mask, num_featuers_nonzero, degrees, neigh_info)
    print('test acucracy: ', accu)
#Test graphsage maxpool finish


#Test graphsage meanpool
elif model_name == 'graphsage_meanpool':
    degrees = data['degrees']
    row = data['row']
    col = data['col']

    model = GraphSageMeanPool(
        hidden_num = 1, hidden_dim = [hidden_dim],
        **addi_parameters,
        learning_rate = learning_rate, epochs = epochs,
        weight_decay = weight_decay, early_stopping = early_stopping,
        activation_func = activation_func,
        dropout_prob = dropout_prob,
        bias = bias,
        optimizer = optimizer,
        name='GraphSage',
        transform_size = [hidden_dim, addi_parameters['output_dim']]
    )

    sess = tf.Session()

    model.train(sess, undirected, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero, degrees, row, col)


    accu, time_used = model.test(sess, undirected, features, y_test, test_mask, num_featuers_nonzero, degrees, row, col)
    print('test acucracy: ', accu)
#Test graphsage meanpool finish


else:
	pass
