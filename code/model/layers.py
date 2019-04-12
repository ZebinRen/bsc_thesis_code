import tensorflow as tf

class BaseLayer(object):
    def __init__(self,
                 input_shape, output_shape,
                 activation,
                 name,
                 dropout_prob = None,
                 bais = False,
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
        
        #Create variables
        with tf.variable_scope(self.name):
    
    def run(self, inputs):

