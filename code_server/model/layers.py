import tensorflow as tf
import scipy.sparse as sp
from .layer_utils import *
import numpy as np

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
        self.dropout_prob = dropout_prob
        self.bias = bias
        self.sparse = sparse
        self.weight_decay_vars = []

        self.input_dim = input_dim
        self.output_dim = output_dim


        #If dropout_prob is assigned a value
        if self.dropout_prob:
            #Check if the type of dropout_prob is legal
            if type(self.dropout_prob) is type(1.0):
                pass
            else:
                print("droupout_prob is: ", self.dropout_prob)
                raise Exception('Invalid type for dropout.')

            #Check if the value is legal
            if 0.0 < self.dropout_prob < 1.0:
                pass
            else:
                print("droupout_prob is: ", self.dropout_prob)
                raise Exception('Invalid value for droupout.')
            

    #This function is invocked by the object name
    def __call__(self, inputs, num_features_nonzero):
        with tf.name_scope(self.name):   
            return self.run(inputs, num_features_nonzero)
    
    def run(self, inputs):
        '''
        Run the layers
        This will bulid the graph
        This function will connect the prev layer's output as input
        Then provide current layer's output as return value
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
                 sparse = False,):
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
            self.weights = glort_init([input_dim, output_dim], name = 'weights')
            self.weight_decay_vars.append(self.weights)
        
        #If bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')
                self.weight_decay_vars.append(self.bias)    

    def run(self, inputs, num_features_nonzero = None):
        '''
        Inputs are features, Since the feateure map will change through the network
        The symmertic normalized Laplacian matrix at the first layer
        Then the convoluted matrix in the following layers
        '''
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)
        
        #Do convolution
        output = graph_conv(inputs, self.adjancy, self.weights, self.sparse)

        #bias
        if self.bias != None:
            output += self.bias

        #activation
        return self.activation_func(output)


class DenseLayer(BaseLayer):
    '''
    Dense Layer
    '''
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False):
        super(DenseLayer, self).__init__(
                                            input_dim, output_dim,
                                            activation_func,
                                            name,
                                            dropout_prob,
                                            bias,
                                            sparse
                                            )

        self.adjancy = adjancy

        #Define layer's variables
        with tf.variable_scope(self.name + '_var'): 
            self.weights = glort_init([input_dim, output_dim], name = 'weights')
            self.weight_decay_vars.append(self.weights)


            #if bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')
                self.weight_decay_vars.append(self.bias)


    def run(self, inputs, num_features_nonzero):
        '''
        Inputs are features or the output passed by the previous layer
        This will connect each layers into one compution graph
        '''


        #Note, sparse drop is not implemented
        #Since we assume that no dropout is implemented and the output of a layer is dense matrix
        #Drop out to input can be implemented befor it is feeded to the train function 
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)


        #Do the calculation
        if self.sparse:
            output = tf.sparse_tensor_dense_matmul(inputs, self.weights)
        else:
            output = tf.matmul(inputs, self.weights)


        #bias
        if self.bias != None:
            output += self.bias

        #acitvation
        return self.activation_func(output)



class FirstChebLayer(BaseLayer):
    '''
    First Cheb Layer
    Note that the adjancy matrix is not normalized
    '''
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False):
        super(FirstChebLayer, self).__init__(
                                            input_dim, output_dim,
                                            activation_func,
                                            name,
                                            dropout_prob,
                                            bias,
                                            sparse
                                            )

        self.adjancy = adjancy

        #Define layer's variables
        with tf.variable_scope(self.name + '_var'): 
            self.weights_0 = glort_init([input_dim, output_dim], name = 'weights_0')
            self.weight_decay_vars.append(self.weights_0)
            self.weights_1 = glort_init([input_dim, output_dim], name = 'weights_1')
            self.weight_decay_vars.append(self.weights_1)


            #if bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')
                self.weight_decay_vars.append(self.bias)


    def run(self, inputs, num_features_nonzero):
        '''
        Inputs are features or the output passed by the previous layer
        This will connect each layers into one compution graph
        '''


        #Note, sparse drop is not implemented
        #Since we assume that no dropout is implemented and the output of a layer is dense matrix
        #Drop out to input can be implemented befor it is feeded to the train function 
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)



        if self.sparse:
            XW_0 = tf.sparse_tensor_dense_matmul(inputs, self.weights_0)
            XW_1 = tf.sparse_tensor_dense_matmul(inputs, self.weights_1)
            second_part = tf.sparse_tensor_dense_matmul(self.adjancy, XW_1)
            output = tf.add(XW_0, second_part)
        else:
            XW_0 = tf.matmul(inputs, self.weights_0)
            XW_1 = tf.matmul(inputs, self.weights_1)
            second_part = tf.sparse_tensor_dense_matmul(self.adjancy, XW_1)
            output = tf.add(XW_0, second_part)


        #bias
        if self.bias != None:
            output += self.bias

        #acitvation
        return self.activation_func(output)

'''
class GraphAttentionLayer(BaseLayer):
    #
    #Graph Attention Layer
    #attention head is the number of attention heads, default = 1
    #aggregate have two values: concate, ave
    #concate will concate all the ouputs, used in hidded layers
    #and ave will average the outputs, used in the output layer
  
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 total_nodes,
                 attention_head = 1,
                 dropout_prob = None,
                 bias = False,
                 sparse = False,
                 aggregate_mode = 'concate'):
        super(GraphAttentionLayer, self).__init__(
                                            input_dim, output_dim,
                                            activation_func,
                                            name,
                                            dropout_prob,
                                            bias,
                                            sparse
                                            )

        self.adjancy = adjancy
        self.attention_head = attention_head
        self.aggregate_mode = aggregate_mode
        self.total_nodes = total_nodes

        #Define layer's variables
        #The number of weights and attention is equal to the number of attention heads
        self.weights = []
        self.attention = []
        self.pre_weights = []
        with tf.variable_scope(self.name + '_var'): 
            #Add weights and attention
            for i in range(self.attention_head):
                self.weights.append(glort_init([output_dim, input_dim], name = 'weights_' + str(i)))
                self.pre_weights.append(glort_init([output_dim, input_dim], name = 'pre_weights_' + str(i)))
                self.attention.append(tf.zeros([output_dim*2], name = 'attention_' + str(i)))

            
            #if bias is used
            if self.bias:
                if aggregate_mode == 'concate':
                    self.bias = tf.zeros([output_dim*attention_head], name = 'bias')
                elif aggregate_mode == 'ave':
                    self.bias = tf.zeros([output_dim], name = 'bias')
                else:
                    raise "Invalid value for aggregate_mode"
            

    def run(self, inputs, num_features_nonzero):
        #
        #Inputs are features or the output passed by the previous layer
        #This will connect each layers into one compution graph
        #


        #Note, sparse drop is not implemented
        #Since we assume that no dropout is implemented and the output of a layer is dense matrix
        #Drop out to input can be implemented befor it is feeded to the train function 
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)


        #Do the calculation
        WX_T = []
        if self.sparse:
            #Compute a list of WX_T, X is a sparse tensor
            #So, compute (WX_T)_T = XW_T
            #Then transpose it back
            for i in range(self.attention_head):
                mid_value = tf.sparse_tensor_dense_matmul(inputs, tf.transpose(self.weights[i]))
                WX_T.append(tf.transpose(mid_value))

        else:
            #Directly compute WX_T
            for i in range(self.attention_head):
                WX_T.append(tf.matmul(self.weights[i], tf.transpose(inputs)))

        #Compute a_T[Wh_i||Wh_j]
        #Note the concatention is calculted later
        #AWX_T_1 is the front part of the multiplication
        #AWX_T_2 is the back part of the multiplicationh
        #each of them is a #NODE vector

        AWX_T_1 = []
        AWX_T_2 = []

        for i in range(self.attention_head):
            AWX_T_1.append(tf.einsum('i,kj->j', self.attention[i][:self.output_dim], WX_T[i]))
            AWX_T_2.append(tf.einsum('i,kj->j', self.attention[i][self.output_dim+1: ], WX_T[i]))

        #Compute attention coeffcients
        attention_coffe = []

        for i in range(self.attention_head):
            first = tf.stack(AWX_T_1[i]*self.total_nodes)
            first = tf.transpose(first)
            second = tf.stack([AWX_T_2[i]] * self.total_nodes)
            attention_coffe.append(tf.add(first, second))

        #Mask the coeffcient matrix by adjancy matrix
        #And use LeakyRelu, 0.2
        #compute softmax over row
        for i in range(self.attention_head):     
            attention_coffe[i] = mask_by_adj(attention_coffe[i], self.adjancy)
            attention_coffe[i] = tf.nn.leaky_relu(attention_coffe[i], alpha = 0.2)
            attention_coffe[i] = tf.nn.softmax(attention_coffe[i])

        #Compute the predct
        #For each attention coffecients matrix A and weight matrix W
        #output = WX_TA_T
        output = []
        mid_value = None
        for i in range(self.attention_head):
            if self.sparse:
                mid_value = tf.sparse_tensor_dense_matmul(inputs, tf.transpose(self.pre_weights[i]))
                mid_value = tf.transpose(mid_value)
            else:
                mid_value = tf.matmul(self.pre_weights[i], tf.transpose(inputs))

            output.append(tf.matmul(mid_value, tf.transpose(attention_coffe[i])))
            

        if self.aggregate_mode == 'concate':
            #Concate the matrix
            output = tf.concat(output, axis = 0)

        elif self.aggregate_mode == 'ave':
            #Ave over output_matrix
            len_output = len(output)
            output = tf.add_n(output)
            output = output/len_output
        else:
            raise "aggregate_mode has a invalid value"

        #Transpose the output
        output = tf.transpose(output)

        #bias
        if self.bias != None:
            output += self.bias

        #acitvation
        return self.activation_func(output)
'''


class GraphAttentionLayer(BaseLayer):
    #
    #Graph Attention Layer
    #attention head is the number of attention heads, default = 1
    #aggregate have two values: concate, ave
    #concate will concate all the ouputs, used in hidded layers
    #and ave will average the outputs, used in the output layer
  
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 total_nodes,
                 attention_head = 1,
                 dropout_prob = None,
                 bias = False,
                 sparse = False,
                 row = None,
                 col = None,
                 indices = None,
                 num_nodes = None,
                 aggregate_mode = 'concate'):
        super(GraphAttentionLayer, self).__init__(
                                            input_dim, output_dim,
                                            activation_func,
                                            name,
                                            dropout_prob,
                                            bias,
                                            sparse
                                            )

        self.adjancy = adjancy
        self.attention_head = attention_head
        self.aggregate_mode = aggregate_mode
        self.total_nodes = total_nodes
        self.row = row
        self.col = col
        self.num_nodes = num_nodes
        self.indices = indices

        #Define layer's variables
        #The number of weights and attention is equal to the number of attention heads
        self.weights = []
        self.att_i_weights = []
        self.att_j_weights = []
        with tf.variable_scope(self.name + '_var'): 
            #Add weights and attention
            for i in range(self.attention_head):
                tmp_weight = glort_init([self.input_dim, self.output_dim], name = 'weights_' + str(i))
                self.weights.append(tmp_weight)
                tmp_i = tf.zeros([output_dim, 1], name = 'attention_' + str(i))
                tmp_j = tf.zeros([output_dim, 1], name = 'attention_' + str(i))
                self.att_i_weights.append(tmp_i)
                self.att_j_weights.append(tmp_j)
                self.weight_decay_vars.append(tmp_weight)
                self.weight_decay_vars.append(tmp_i)
                self.weight_decay_vars.append(tmp_j)


            
            #if bias is used
            if self.bias:
                if aggregate_mode == 'concate':
                    self.bias = tf.zeros([output_dim*attention_head], name = 'bias')
                elif aggregate_mode == 'ave':
                    self.bias = tf.zeros([output_dim], name = 'bias')
                else:
                    raise "Invalid value for aggregate_mode"
            

    def run(self, inputs, num_features_nonzero):
        #
        #Inputs are features or the output passed by the previous layer
        #This will connect each layers into one compution graph
        #


        #Note, sparse drop is not implemented
        #Since we assume that no dropout is implemented and the output of a layer is dense matrix
        #Drop out to input can be implemented befor it is feeded to the train function
        def lrelu(x, alpha):
           return tf.nn.relu(x) - alpha * tf.nn.relu(-x) 

        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)


        #Do the calculation
        trans_feature = []
        if self.sparse:
            for i in range(self.attention_head):
                tmp_value = tf.sparse_tensor_dense_matmul(inputs, self.weights[i])
                trans_feature.append(tmp_value)

        else:
            #Directly compute WX_T
            for i in range(self.attention_head):
                trans_feature.append(tf.matmul(inputs, self.weights[i]))

        #Compute a_T[Wh_i||Wh_j]
        #Note the concatention is calculted later
        #AWX_T_1 is the front part of the multiplication
        #AWX_T_2 is the back part of the multiplicationh
        #each of them is a #NODE vector
        att_i_list = []
        att_j_list = []
        for i in range(self.attention_head):
            tmp_i = tf.matmul(trans_feature[i], self.att_i_weights[i])
            tmp_j = tf.matmul(trans_feature[i], self.att_j_weights[i])
            tmp_i = tf.contrib.layers.bias_add(tmp_i)
            tmp_j = tf.contrib.layers.bias_add(tmp_j)
            att_i_list.append(tmp_i)
            att_j_list.append(tmp_j)

        att_coffe_list = []
        for i in range(self.attention_head):
            att_coffe = tf.gather(att_i_list[i], self.row) + \
                        tf.gather(att_j_list[i], self.col)

            att_coffe = tf.squeeze(att_coffe)
            att_coffe_list.append(att_coffe)

        att_coffe_mat_list = []

        for i in range(self.attention_head):
            att_coffe_mat = tf.SparseTensor(
            indices=self.indices,
            values=lrelu(att_coffe_list[i], alpha=0.2), 
            dense_shape=(self.num_nodes, self.num_nodes)
            )
            att_coffe_mat = tf.sparse_softmax(att_coffe_mat)
            att_coffe_mat_list.append(att_coffe_mat)


        output = []
        for i in range(self.attention_head):
            cur_output = tf.sparse_tensor_dense_matmul(att_coffe_mat_list[i], trans_feature[i])
            if self.bias != None:
                cur_output = tf.contrib.layers.bias_add(cur_output)
            output.append(cur_output)

        if self.aggregate_mode == 'concate':
            #Concate the matrix
            output = tf.concat(output, axis = 1)

        elif self.aggregate_mode == 'ave':
            #Ave over output_matrix
            len_output = len(output)
            output = tf.add_n(output)
            output = output/len_output
        else:
            raise "aggregate_mode has a invalid value"


        #acitvation
        return self.activation_func(output)




class DiffusionLayer(BaseLayer):
    '''
    First Cheb Layer
    Note that the adjancy matrix is not normalized
    '''
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False,
                 num_nodes = False,
                 hops = 3):
        super(DiffusionLayer, self).__init__(
            input_dim, output_dim,
            activation_func,
            name,
            dropout_prob,
            bias,
            sparse
            )

        self.adjancy = adjancy
        self.hops = hops
        self.num_nodes = num_nodes

        #Define layer's variables
        with tf.variable_scope(self.name + '_var'): 
            self.weight_c = glort_init([self.hops, input_dim], name = 'weight_c')
            self.weight_d = glort_init([self.hops * input_dim, output_dim], name = 'weight_d')
            self.weight_decay_vars.append(self.weight_c)
            self.weight_decay_vars.append(self.weight_d)


            #if bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')
                self.weight_decay_vars.append(self.bias)


    def run(self, inputs, num_features_nonzero):
        '''
        Inputs are features or the output passed by the previous layer
        This will connect each layers into one compution graph
        '''


        #Note, sparse drop is not implemented
        #Since we assume that no dropout is implemented and the output of a layer is dense matrix
        #Drop out to input can be implemented befor it is feeded to the train function 
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
                inputs = tf.sparse_to_dense(inputs.indices, inputs.dense_shape, inputs.values)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)

        #Shape Nt * H * Nt
        pt_series = create_power_series(self.adjancy, self.hops, sparse=True)


        #compute P_X
        #dim (Nt*H*Nt) * (Nt*F)
        pt_series.set_shape((self.num_nodes, self.hops, self.num_nodes))
        inputs.set_shape((self.num_nodes, self.input_dim))
        P_X = tf.einsum('ijk,kl->ijl', pt_series , inputs)

        #compute W_c (element-wise product) P_X
        WPX = P_X * self.weight_c
        WPX = tf.nn.tanh(WPX)

        #flatten, let Z be a two-dim matrix Nt*(H*F)
        Z = tf.contrib.layers.flatten(WPX)
        print(Z)
        print(self.weight_d)
        print(self.bias)

        output = tf.matmul(Z, self.weight_d)

       
        #bias
        if self.bias != None:
            output += self.bias

        #acitvation
        return self.activation_func(output)




'''
class SpectralCNNLayer(BaseLayer):
    #Dense Layer
    def __init__(self,
                 eigenvalue_matrix,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False,
                 total_nodes = None):
        super(SpectralCNNLayer, self).__init__(
                                            input_dim, output_dim,
                                            activation_func,
                                            name,
                                            dropout_prob,
                                            bias,
                                            sparse
                                            )

        self.ei_mat = eigenvalue_matrix
        self.total_nodes = total_nodes

        #Define layer's variables
        with tf.variable_scope(self.name + '_var'):
            self.weights_list = []
            for i in range(output_dim): 
                self.weights_list.append(glort_init([total_nodes, input_dim], name = 'weights' + str(i)))
            
            #Stack self.weights as a single three-dim vector
            self.weights = tf.stack(self.weights_list, axis=0)


            #if bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')


    def run(self, inputs, num_features_nonzero):
        #Inputs are features or the output passed by the previous layer
        #This will connect each layers into one compution graph


        #Note, sparse drop is not implemented
        #Since we assume that no dropout is implemented and the output of a layer is dense matrix
        #Drop out to input can be implemented befor it is feeded to the train function 
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)


        #Do the calculation
        if self.sparse:
            X_TV = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(inputs), self.ei_mat)
            V_TX = tf.transpose(X_TV)
        else:
            V_TX = tf.matmul(tf.transpose(self.ei_mat), inputs)

        WV_TX = V_TX * self.weights

        output = tf.einsum('jk,ikl->ijl', self.ei_mat , WV_TX)

        #Output shape is output_dim, total_nodes, input_dim
        #Add over input features
        #output_dim, total_nodes
        #Add over total_nodes
        output = tf.reduce_sum(output, 2)
        output = tf.transpose(output)


        #bias
        if self.bias != None:
            output += self.bias

        #acitvation
        return self.activation_func(output)
'''

class SpectralCNNLayer(BaseLayer):
    '''
    Dense Layer
    '''
    def __init__(self,
                 eigenvalue_matrix,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False,
                 total_nodes = None):
        super(SpectralCNNLayer, self).__init__(
                                            input_dim, output_dim,
                                            activation_func,
                                            name,
                                            dropout_prob,
                                            bias,
                                            sparse
                                            )

        self.ei_mat = eigenvalue_matrix
        self.total_nodes = total_nodes

        #Define layer's variables
        with tf.variable_scope(self.name + '_var'):
            self.weights=glort_init([input_dim, output_dim], name = 'weights')
            self.weight_decay_vars.append(self.weights)


            #if bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')
                self.weight_decay_vars.append(self.bias)


    def run(self, inputs, num_features_nonzero):
        '''
        Inputs are features or the output passed by the previous layer
        This will connect each layers into one compution graph
        '''


        #Note, sparse drop is not implemented
        #Since we assume that no dropout is implemented and the output of a layer is dense matrix
        #Drop out to input can be implemented befor it is feeded to the train function 
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)

        if self.sparse:
            XW = tf.sparse_tensor_dense_matmul(inputs, self.weights)
        else:
            XW = tf.matmul(inputs, self.weights)
        
        V_TXW = tf.matmul(tf.transpose(self.ei_mat), XW)
        
        '''
        #Do the calculation
        if self.sparse:
            X_TV = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(inputs), self.ei_mat)
            V_TX = tf.transpose(X_TV)
        else:
            V_TX = tf.matmul(tf.transpose(self.ei_mat), inputs)

        V_TXW = tf.matmul(V_TX, self.weights)
        '''
        output = tf.matmul(self.ei_mat , V_TXW)

        #Output shape is output_dim, total_nodes, input_dim
        #Add over input features
        #output_dim, total_nodes
        #Add over total_nodes

        #bias
        if self.bias != None:
            output += self.bias

        #acitvation
        return self.activation_func(output)



class ChebLayer(BaseLayer):
    '''
    ChebLayer: Compute cheblayer
    '''
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False,
                 poly_order = None,
                 ):
        super(ChebLayer, self).__init__(
                input_dim, output_dim,
                activation_func,
                name,
                dropout_prob,
                bias,
                sparse
                )

        self.adjancy = adjancy
        self.poly_order = poly_order

        #Compute the cheb polynimials
        self.series = self.adjancy

        #Define layer's variables
        self.weights_list = []
        with tf.variable_scope(self.name + '_var'):
            for i in range(self.poly_order):
                tmp = glort_init([self.input_dim, self.output_dim], name = 'weights' + str(i))
                self.weights_list.append(tmp)
                self.weight_decay_vars.append(tmp)                    

            #if bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')
                self.weight_decay_vars.append(self.bias)

    def run(self, inputs, num_features_nonzero):
        '''
        Inputs are features or the output passed by the previous layer
        This will connect each layers into one compution graph
        '''


        #Note, sparse drop is not implemented
        #Since we assume that no dropout is implemented and the output of a layer is dense matrix
        #Drop out to input can be implemented befor it is feeded to the train function 
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)

        
        #compute XW
        if self.sparse:
            for i in range(self.poly_order):
                self.weights_list[i] = tf.sparse_tensor_dense_matmul(inputs, self.weights_list[i])
        else:
            for i in range(self.poly_order):
                self.weights_list[i] = tf.matmul(inputs, self.weights_list[i])

        #compute TXW
        TXW = []
        for i in range(self.poly_order):
            TXW.append(tf.sparse_tensor_dense_matmul(self.series[i], self.weights_list[i]))

        output = tf.add_n(TXW)


        #bias
        if self.bias != None:
            output += self.bias

        #acitvation
        return self.activation_func(output)



class MeanLayer(BaseLayer):
    '''
    MeanLayer, used in GraphSage
    '''
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False,
                 degrees = None):
        super(MeanLayer, self).__init__(
            input_dim, output_dim,
            activation_func,
            name,
            dropout_prob,
            bias,
            sparse
            )

        self.adjancy = adjancy
        self.degrees = degrees

        
        #Define layers' variable
        with tf.variable_scope(self.name + '_var'):
            self.weights = glort_init([input_dim, output_dim], name = 'weights')
            self.weight_decay_vars.append(self.weights)
            self.skip_weights = glort_init([self.input_dim, output_dim], name = 'skip_weights')
            self.weight_decay_vars.append(self.skip_weights)
        
        #If bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')
                self.weight_decay_vars.append(self.bias)    
    def run(self, inputs, num_features_nonzero = None):
        '''
        Inputs are features, Since the feateure map will change through the network
        The symmertic normalized Laplacian matrix at the first layer
        Then the convoluted matrix in the following layers
        '''
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)
        
        #Multiply it by weight matrix
        if self.sparse:
            x = tf.sparse_tensor_dense_matmul(inputs, self.weights)
        else:
            x = tf.matmul(inputs, self.weights)

        #multiply it by adjancy matrix
        x = tf.sparse_tensor_dense_matmul(self.adjancy, x)

        #Divide by degree
        x = x/self.degrees


        #Compute skipped features
        if self.sparse:
            skipped_features = tf.sparse_tensor_dense_matmul(inputs, self.skip_weights)
        else:
            skipped_features = tf.matmul(inputs, self.skip_weights)

        x = x + skipped_features

        if self.bias != None:
            x += self.bias

        #normalizer = tf.norm(x, ord=2, axis=1, keepdims=True)
        #normalizer = tf.clip_by_value(normalizer, clip_value_min=1.0, clip_value_max=np.inf)
        #print('###', normalizer)
        #exit()
        #output = x / normalizer
        output = x

        #activation
        return self.activation_func(output)



class MeanPoolLayer(BaseLayer):
    '''
    MeanPoolLayer, used in GraphSage
    '''
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False,
                 degrees = None,
                 row = None,
                 col = None,
                 transform_size = None,
                 num_nodes = None):
        super(MeanPoolLayer, self).__init__(
            input_dim, output_dim,
            activation_func,
            name,
            dropout_prob,
            bias,
            sparse
            )

        self.adjancy = adjancy
        self.degrees = degrees
        self.row = row
        self.col = col
        self.num_nodes = num_nodes
        self.transform_size = transform_size

        
        #Define layers' variable
        with tf.variable_scope(self.name + '_var'):
            self.agg_weights = glort_init([input_dim, transform_size], name = 'agg_weights')
            self.weight_decay_vars.append(self.agg_weights)
            self.output_weights = glort_init([transform_size, output_dim], name = 'ouptut_weights')
            self.weight_decay_vars.append(self.output_weights)
            self.skip_weights = glort_init([self.input_dim, output_dim], name = 'skip_weights')
            self.weight_decay_vars.append(self.skip_weights)
           
        #If bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')
                self.weight_decay_vars.append(self.bias)    
    def run(self, inputs, num_features_nonzero = None):
        '''
        Inputs are features, Since the feateure map will change through the network
        The symmertic normalized Laplacian matrix at the first layer
        Then the convoluted matrix in the following layers
        '''
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)
        
        #transform features
        if self.sparse:
            trans_features = tf.sparse_tensor_dense_matmul(inputs, self.agg_weights)
        else:
            trans_features = tf.matmul(inputs, self.agg_weights)

        #acitvate
        trans_features = tf.nn.relu(trans_features)

        agg_features = tf.gather(trans_features, self.row)
        #scatter_add_tensor(agg_features, self.col, out_shape=[num_nodes, agg_transform_size])
        col = tf.expand_dims(self.col, -1)
        agg_features = tf.scatter_nd(col, agg_features, shape=[self.num_nodes, self.transform_size])

        #resize
        agg_features = agg_features/self.degrees

        x = tf.matmul(agg_features, self.output_weights)
        

        #Compute skipped features
        if self.sparse:
            skipped_features = tf.sparse_tensor_dense_matmul(inputs, self.skip_weights)
        else:
            skipped_features = tf.matmul(inputs, self.skip_weights)

        x = x + skipped_features

        if self.bias != None:
            x += self.bias

        #normalizer = tf.norm(x, ord=2, axis=1, keepdims=True)
        #normalizer = tf.clip_by_value(normalizer, clip_value_min=1.0, clip_value_max=np.inf)
        #print('###', normalizer)
        #exit()
        #output = x / normalizer
        output = x

        #activation
        return self.activation_func(output)

class MaxPoolLayer(BaseLayer):
    '''
    MaxPoolLayer, used in GraphSage
    '''
    def __init__(self,
                 adjancy,
                 input_dim, output_dim,
                 activation_func,
                 name,
                 dropout_prob = None,
                 bias = False,
                 sparse = False,
                 degrees = None,
                 neigh_info = None,
                 transform_size = None):
        super(MaxPoolLayer, self).__init__(
            input_dim, output_dim,
            activation_func,
            name,
            dropout_prob,
            bias,
            sparse
            )

        self.adjancy = adjancy
        self.degrees = degrees
        self.neigh_info = neigh_info
        self.transform_size = transform_size

        
        #Define layers' variable
        with tf.variable_scope(self.name + '_var'):
            self.agg_weights = glort_init([input_dim, transform_size], name = 'agg_weights')
            self.weight_decay_vars.append(self.agg_weights)
            self.output_weights = glort_init([transform_size, output_dim], name = 'ouptut_weights')
            self.weight_decay_vars.append(self.output_weights)
            self.skip_weights = glort_init([self.input_dim, output_dim], name = 'skip_weights')
            self.weight_decay_vars.append(self.skip_weights)
           
        #If bias is used
            if self.bias:
                self.bias = tf.zeros([output_dim], name = 'bias')
                self.weight_decay_vars.append(self.bias)    
    def run(self, inputs, num_features_nonzero = None):
        '''
        Inputs are features, Since the feateure map will change through the network
        The symmertic normalized Laplacian matrix at the first layer
        Then the convoluted matrix in the following layers
        '''
        if not self.dropout_prob:
            pass

        else:
            if self.sparse:
                inputs = sparse_dropout(inputs, 1 - self.dropout_prob, num_features_nonzero)
            else:
                inputs = tf.nn.dropout(inputs, 1 - self.dropout_prob)
        
        #transform features
        if self.sparse:
            trans_features = tf.sparse_tensor_dense_matmul(inputs, self.agg_weights)
        else:
            trans_features = tf.matmul(inputs, self.agg_weights)

        #acitvate
        trans_features = tf.nn.relu(trans_features)

        #neighbours features
        neighbours_features = tf.gather(trans_features, self.neigh_info)

        #max aggregator
        agg_features = tf.reduce_max(neighbours_features, axis=1)

        #aggregated features
        x = tf.matmul(agg_features, self.output_weights)

        #Compute skipped features
        if self.sparse:
            skipped_features = tf.sparse_tensor_dense_matmul(inputs, self.skip_weights)
        else:
            skipped_features = tf.matmul(inputs, self.skip_weights)

        x = x + skipped_features

        if self.bias != None:
            x += self.bias

        #normalizer = tf.norm(x, ord=2, axis=1, keepdims=True)
        #normalizer = tf.clip_by_value(normalizer, clip_value_min=1.0, clip_value_max=np.inf)
        #print('###', normalizer)
        #exit()
        #output = x / normalizer
        output = x

        #activation
        return self.activation_func(output)
