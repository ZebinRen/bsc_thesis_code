import tensorflow as tf
from .base_model import BaseModel
from .layers import *
from .model_utils import *

class GCN(BaseModel):
    def __init__(self,
         hidden_num, hidden_dim,
         input_dim, output_dim,
         node_num, cate_num,
         learning_rate, epochs,
         weight_decay, early_stopping,
         activation_func, 
         dropout_prob,
         bias, name):
        super(GCN, self).__init__(
            hidden_num, hidden_dim,
            input_dim, output_dim,
            learning_rate, epochs,
            weight_decay, early_stopping,
            name)

        self.total_nodes = input_dim
        self.total_cates = output_dim
        self.activation_func = activation_func
        self.dropout_prob = dropout_prob
        self.bias = bias
  
        #Add placeholders
        #Note, this dictionary is used to create feed dicts
        self.placeholders = {}
        self.placeholders['features'] = tf.sparse_placeholder(tf.float32, shape=(), name='Feature')
        self.placeholders['adj'] = tf.sparse_placeholder(tf.float32, shape=(), name='Adjancy')
        self.placeholders['labels'] = tf.placeholder(tf.int32, )
        self.placeholders['mask'] = tf.placeholder(tf.int32)

        self.adjancy = self.placeholders['adj']
        self.inputs = self.placeholders['features']
        self.label = self.placeholders['labels']
        self.mask = self.placeholders['mask']

        self.build()


    def _add_layers(self):
        for i in range(self.hidden_num + 1):
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
                         self.name + '_' + str(i),
                         self.dropout_prob,
                         self.bias,
                         sparse = sparse_input)
      )
  
    def _loss(self):
        self.loss = 0
        #Regularization

        #loss
        self.loss += masked_softmax_cross_entropy(self.outputs, self.labels, self.mask) 

    def train(self, sess, adj, features, label, mask):
        '''
        Train the model
        '''
        #Loss: Saves a list of the loss
        loss_list = []
        #Construct the feed dict
        feed_dict = {
          self.adjancy: adj,
          self.features: features,
          self.label: label,
          self.mask: mask
        }
        
        sess.run(tf.global_variables_initializer())

        #Train precedure
        for epoch in self.epochs:

            loss = sess.run(self.loss, feed_dict=feed_dict)
            loss_list.append(loss)

            #Debug
            print('epochs: ', epoch, 'loss: ', loss)
            #Test early stopping


    

    def predict(self, sess, adj, features, label, mask):
        feed_dict = {
            self.adjancy: adj,
            self.features: features,
            self.label: label,
            self.mask: mask
        }

        outputs = sess.run(self.outputs, feed_dict=feed_dict)
        
        #mask
        mask = tf.cast(mask, dtype=tf.float32)
        outputs = mask * outputs

        return outputs





    
