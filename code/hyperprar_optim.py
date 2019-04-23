from model.gcn import gcn
from model.mlp import mlp

learning_rate = [0.1, 0.07, 0.05, 0.03, 0.01, 0.007, 0.005, 0.003, 0.001, 0.0005, 0.0001]
epochs = 400
weight_decay = [5e-1, 1e-1, 1e-2, 7e-3, 5e-3, 3e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 5e-5, 1e-5]
early_stopping = [20, 40, 60, 80, 100]
activation_func = [tf.nn.relu
dropout_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
bias = True
hidden_dim = [12, 16, 20, 24, 28, 32, 36, 40]
optimizer = [tf.train.AdamOptimizer