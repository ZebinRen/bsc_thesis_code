'''
This script is used to debug te model
Write the code in a new file to test the models
'''

import tensorflow as tensorflow
import create_dataset from load_data
from utils import *


directed, undirected, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,\
info = create_input('./data', 'citeseer', 0, 500, 500, None)

#preprocess the adjancy matrix for GCN
directed = symmertic_normalized_laplacian(directed)
undirected = symmertic_normalized_laplacian(undirected)

#preprocess features



