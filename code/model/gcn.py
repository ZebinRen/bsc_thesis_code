import tensorflow as tf
from base_model import BaseModel
from layers import *

class GCN(Model):
    def __init__(self,
				 hidden_num, hidden_dim,
				 input_dim, output_dim,
				 total_nodes, total_cates,
				 learning_rate, epochs,
         weight_decay, early_stopping,
				 activation_func, 
				 dropout_prob,
				 bias,
				 name, dataset=None):
        super(GCN, self).__init__(
						hidden_num, hidden_dim,
						input_dim, output_dim,
						leaning_rate, epochs,
						weight_decay, early_stopping,
						name)

		    self.total_nodes = total_nodes
		    self.total_cates = total_cates
		    self.activation_func = activation_func
		    self.dropout_prob = dropout_prob
		    self.bias = bias
	
		#Add placeholders
		#Note, this dictionary is used to create feed dicts
		self.placeholders = {}
		self.placeholders['features'] = tf.sparse_placeholder(tf.float32, shape=(), name='Feature')
		self.placeholders['adj'] = tf.sparse_placeholder(tf.float32, shape=(), name='Adjancy')
		self.placeholders['labels'] = tf.placeholder(tf.int32, )
		self.placeholders['label_mask'] = tf.placeholder(tf.int32)

		self.adjancy = self.placeholders['adj']
		self.inputs = self.placeholders['features']
		self.labels = self.placeholder['labels']
		self.label_mask = self.placeholder['label_mask']

		build()


    def _add_layers(self):
		for i in range(hidden_num + 1):
			#only the input layer provides a sparse input matrix
			if 0 == i:
				sparse_input = True
			else:
				sparse_input = False
			#each layer has a variable scope
			self.layers.append(
				  GraphConvLayer(self.adjancy,
							           self.hidden_dim[i], self.hidden_dim[i+1],
							           self.activation_func,
							           '_GCL_' + str(i),
							           self.dropout_prob,
							           self.bias,
							           sparse = sparse_input)
			)
	
	  def _loss(self):

	
	  def train(self)
		

    