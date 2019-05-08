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
from hyperpara_optim import *
#from load_data import create_input

learning_rate = 0.01 #0.01
epochs = 4000
weight_decay = 0.003 #5e-1
early_stopping = 100 #500
activation_func = tf.nn.relu
dropout_prob = 0.2 #0.5
bias = True
hidden_dim = 16
optimizer = tf.train.AdamOptimizer

'''
directed, undirected, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,\
info = create_input('./data', 'citeseer', 1, 230, 500, None)

#preprocess the adjancy matrix for GCN
directed = symmetric_normalized_laplacian(directed)
undirected = symmetric_normalized_laplacian(undirected)

#preprocess features
norm_features = row_normalized(features)

#information
nodes = directed.shape[0]
input_dim = features.shape[1]
output_dim = y_train.shape[1]

directed = create_load_sparse(directed)
undirected = create_load_sparse(undirected)
features = create_load_sparse(features)
'''
data, addi_parameters = create_input('gat', './data', 'citeseer', 1, 230, 500, None)
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

'''
#Test GCN



#directed = pre_out * (1./(1- dropout_prob))
'''
'''

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


accu = model.test(sess, undirected, features, y_test, test_mask, num_featuers_nonzero)
print('test acucracy: ', accu)

#Test GCN finish
'''


'''
#Test MLP

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


accu = model.test(sess, directed, features, y_test, test_mask, num_featuers_nonzero)
print('test acucracy: ', accu)

#Test MLP finish
'''

'''
#Test first cheb

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


accu = model.test(sess, directed, features, y_test, test_mask, num_featuers_nonzero)
print('test acucracy: ', accu)
#Test first cheb finish
'''


#Test GAT

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

model.train(sess, directed, features, y_train, y_val, train_mask, val_mask, num_featuers_nonzero)


accu = model.test(sess, directed, features, y_test, test_mask, num_featuers_nonzero)
print('test acucracy: ', accu)

#TEST GAT FINISH
