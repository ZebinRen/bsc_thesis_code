import tensorflow as tf
from layer_utils import *

class BaseLayer(object):
    def __init__(self,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False
                 ):
        #Initialize some variables
        self.name = name
        self.activation_func = activation_func


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
	A = D^-0.5 L D^-0.5 (Renomalized Laplacian)
    X is the feature matrix
    NOTE: There is no bias or dropout in the orgin model
    
    '''
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False):
        super(GraphConvLayer, self).__init__(
                                            input_dim, output_dim,
                                            activation_func,
                                            name,
                                            dropout_prob,
                                            bias,
                                            sparse
                                            )
        self.adjancy = adjancy
        
        #Define layers' variable
	with tf.variable_scope(self.name + '_var'):
	    self.weights = glorot_init([input_dim, output_dim], name = 'weights')
	    
	    #If bias is used
	    if self.bias:
		self.bias = zeros_inin([output_dim], name = 'bias')
    
    def run(self, inputs):
        '''
	Inputs are features, Since the feateure map will change through the network
	The symmertic normalized Laplacian matrix at the first layer
	Then the convoluted matrix in the following layers
	'''
        if not self.dropout
	    if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.droupout_prob)
            else:
		x = tf.nn.dropout(inputs, 1 - self.dropout_prob)
		
        #Do convolution
        graph_conv(inputs, self.adjancy, self.weights, True)

        #bias
        if self.bias != None:
            output += self.bias

        #activation
        return self.activation_func(output)

        


