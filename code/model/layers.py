import tensorflow as tf
from layer_utils import *

class BaseLayer(object):
    def __init__(self,
                 input_shape, output_shape,
                 activation,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False
                 ):
        #Initialize some variables
        self.name = name
        self.activation = activation

        #If dropout_prob is assigned a value
        if not dropout_prob:
            #Check if the type of dropout_prob is legal
            if type(dropout_prob) is type(1.0):
                pass
            else:
                raise Exception('Invalid type for dropout.')

            #Check if the value is legal
            if 0.0 < dropout_prob < 1.0:
                pass
            else:
                raise Exception('Invalid value for droupout.')
            
            self.dropout_prob = dropout_prob

        else:
            self.dropout_prob = None
        
        
    
    def run(self, inputs):
        '''
        '''
        raise NotImplementedError

class GraphConvLayer(BaseLayer):
    '''
    Two layer GCN
    Semi-Supervised Classfication with Graph Convolution Networks, Kipf
    Model:
    	Z = f(X, A) = softmax(A RELU(AXW(0))W(1))
	A = D^-0.5 A D^-0.5 (Renomalized Laplacian)
    NOTE: There is no bias or dropout in the orgin model
    '''
    def __init__(self,
                 input_shape, output_shape,
                 activation,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False):
        super(GraphConvLayer, self).__init__(
                                            input_shape, output_shape,
                                            activation,
                                            name,
                                            dropout_prob,
                                            bias,
                                            sparse
                                            )
        
        #Define layers' variable
        with tf.variable_scope(self.name + '_var')
    
    def run(self)

