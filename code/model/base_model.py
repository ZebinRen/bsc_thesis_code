import tensorflow as tf
import os
_MODEL_NAME = []

class BaseModel(object):
    '''
    Base model for all models
    '''
    def __init__(self, 
                 hidden_num, hidden_dim,
                 input_dim, output_dim,
                 learning_rate, epochs,
                 weight_decay, early_stopping,
                 name):
        '''
        Create model
        '''
        # Each model should have a unique name in one run
        if name in _MODEL_NAME:
            print('Model name: ', name, 'is used.')
            exit()
    
        self.name = name

        #Here are some parameters for training
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        
        #Assign input and output dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        #A list of layers in the model
        self.layers = []
        #The number of hidded layers
        #The first layer is the input layer, the last layer is the output layer
        #So, hidden_num = len(layers) - 1
        self.hidden_num = hidden_num
        #The dimension of hidded layers' output
        #The first element is input_dim, the last element is output_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim = [input_dim] + self.hidden_dim
        self.hidden_dim.append(self.output_dim)

        #variables used in the model
        self.vars = {}
        #This is a list of outputs of each layers
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = None

        self.optimizer = None
        self.opt_op = None


    def _add_layers(self):
        '''
        Create layers
        It should be defined in the speicfic model class
        '''
        raise NotImplementedError

    def _loss(self):
        '''
        Loss function, not defined here
        It should be defined in the specific model class
        '''
        raise NotImplementedError

    def _accuracy(self):
        '''
        Compute accuracy
        '''
        raise NotImplementedError
    
    def build(self):
        '''
        Build models
        '''
        
        # Create layers, in variable scope: name
        with tf.variable_scope(self.name):
            self._add_layers()
        
        # Connect inputs and layers
        self.activations.append(self.inputs)
        for each_layer in self.layers:
            #The input is the output of the previous layer
            act = each_layer(self.activations[-1], self.num_features_nonzero)
            self.activations.append(act)
        #output is the last layer's output
        self.outputs = self.activations[-1]
        
        #Get variables
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        #Get variable dictionary

        #Get loss
        self.loss = self._loss()
        self.accuracy = self._accuracy()

        #Assign optimizer
        self.opt_op = self.optimizer.minimize(self.loss)


    def train(self):
        '''
        Train the model
        '''
        raise NotImplementedError
    
    def predict(self, sess):
        '''
        predict the model
        '''
        raise NotImplementedError


    
    
    def draw_graph(self, sess, path, file_name):
        '''
        Use tensorboard to draw the graph
        To use tensorboard
        tensorboard --logdir='PATH' --port 6006
        Then visit http://localhost:6006/
        '''
        writer = tf.summary.FileWriter(os.path.join(path, file_name), sess.graph)
        print(sess.run)
        writer.close()
        