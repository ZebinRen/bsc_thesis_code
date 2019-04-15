import tensorflow as tf
_MODEL_NAME = []

class BaseModel(object):
    '''
    Base model for all models
    '''
    def __init__(self, name, dataset=None):
        '''
        Create model
        '''
        # Each model should have a unique name in one run
        if name in _MODEL_NAME:
            print('Model name: ', name, 'is used.')
            exit()
    
        self.name = name
        self.dataset = dataset

        self.inputs = None
        self.outputs = None

        self.optimizer = None


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
    
    def build(self):
        '''
        Build models
        '''
        # Create layers, in variable scope: name
        with tf.variable_scope(self.name):
            self._add_layers()

        variables = tf.get_collection(tf.GraphKeys)

    def train(self,)
    	pass
    
    def draw_graph(self, path, file_name):
        '''
        Use tensorboard to draw the graph
        '''        


    
    



