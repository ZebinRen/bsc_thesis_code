from model.gcn import GCN
from model.mlp import MLP
from random import randint

import tensorflow as tf

search_parameter = {
    'learning_rate': [0.1, 0.07, 0.05, 0.03, 0.01, 0.007, 0.005, 0.003, 0.001, 0.0005, 0.0001, 0.00005],
    'weight_decay':  [5e-1, 1e-1, 1e-2, 7e-3, 5e-3, 3e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 5e-5, 1e-5],
    'early_stopping': [20, 40, 60, 80],
    'activation_func':  [tf.nn.relu],
    'dropout_prob': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'hidden_dim': [[12], [16], [20], [24], [28], [32], [36]],
    'optimizer': [tf.train.AdamOptimizer]
}

fixed_parameter = {
    'hidden_num': 1,
    'epochs': 500,
    'bias': True,
    'name': 'Search'
}


def random_search(model, dataset_list, dataset_list_val, search_parameter, fixed_parameter, addi_parameter = {}, random_times = 30,
evaluate_times = 3):
    '''
    Use random search to search for hyperparameters
    model is the model class
    dataset_list is a list of dataset used to evaluate the model
    (Note, only the train and val data is visible to the model)
    search is the parameter that needs to be searched
    random_times is the how many search are used
    evaluate_times: How many the model is trained
    '''

    #Best
    best_para = {}
    best_accu = 0

    sess = tf.Session()

    for each_time in range(random_times):

        #Choose hyperparaemter set radomly
        feed_search = {}
        for key in search_parameter:
            values = search_parameter[key]
            feed_search[key] = values[randint(0, len(values)-1)]

        #Test the accuracy
        cur_accu = 0

        for dataset, dataset_val in zip(dataset_list, dataset_list_val):
            for each_evaluate in range(evaluate_times):
                test_model = model(**feed_search, **fixed_parameter, **addi_parameter)
                test_model.train(sess, **dataset)
                tmp_accu,_ = test_model.test(sess, **dataset_val)
                cur_accu += tmp_accu

        cur_accu /= evaluate_times * len(dataset_list)

        #if the current accuracy is higher, change it
        if cur_accu > best_accu:
            best_accu = cur_accu
            best_para = feed_search
        
    #Update best_parameter, add fixed parameter and additional parameters
    best_para.update(fixed_parameter)
    best_para.update(addi_parameter)


    return best_para, best_accu


def desne_search(model, best_para_input, best_accu_input, dataset_list, dataset_list_val, search_parameter, fixed_parameter, addi_parameter = {},
evaluate_times = 3):
    '''
    This function will use the parameters provided in best_para_input
    Each time change one parameter and traverse all the values, choose one 
    And then go for the next parameter
    '''
    #Best
    best_para = best_para_input
    best_accu = best_accu_input
    
    sess = tf.Session()

    for para_key in search_parameter:
        #Get the list of current choosable values
        para_list = search_parameter[para_key]
        
        #Try each parameter in the current parameter list
        for cur_para in para_list:
            #Get the current para dict
            feed_para = best_para
            feed_para[para_key] = cur_para

            cur_accu = 0

            #train the model, get accuracy
            for dataset, dataset_val in zip(dataset_list, dataset_list_val):
                for each_evaluate in range(evaluate_times):
                    print(feed_para, fixed_parameter, addi_parameter)
                    test_model = model(**feed_para, **fixed_parameter, **addi_parameter)
                    test_model.train(sess, **dataset)
                    tmp_accu,_ = test_model.test(sess, **dataset_val)
                    cur_accu += tmp_accu

            cur_accu /= evaluate_times * len(dataset_list)

            #if the current accuracy is higher, change it
            if cur_accu > best_accu:
                best_accu = cur_accu
                best_para = feed_para

    #Update best_parameter, add fixed parameter and additional parameters
    best_para.update(fixed_parameter)
    best_para.update(addi_parameter)


    return best_para, best_accu


def create_train_feed(ori_dataset, model_name, directed = False):
    '''
    create the parameters for train method
    '''
    if directed:
        adj = ori_dataset['directed']
    else:
        adj = ori_dataset['undirected']

    features = ori_dataset['features']
    y_train = ori_dataset['train_label']
    y_val = ori_dataset['val_label']
    num_features_nonzero = features[1].shape

    train_mask = ori_dataset['train_mask']
    val_mask = ori_dataset['val_mask']

    dataset = {
        'adj': adj, 'features': features, 
        'train_label': y_train, 
        'val_label': y_val, 
        'train_mask': train_mask, 
        'val_mask': val_mask,
        'num_features_nonzero': num_features_nonzero
    }

    if 'graphsage' == model_name:
        dataset['degrees'] = ori_dataset['degrees']
    elif 'graphsage_meanpool' == model_name:
        dataset['degrees'] = ori_dataset['degrees']
        dataset['row'] = ori_dataset['row']
        dataset['col'] = ori_dataset['col']
    elif 'graphsage_maxpool' == model_name:
        dataset['degrees'] = ori_dataset['degrees']
        dataset['neigh_info'] = ori_dataset['neigh_info']

    return dataset

def create_test_feed(ori_dataset, model_name, directed = False):
    '''
    Create the parameters for validation
    '''
    if directed:
        adj = ori_dataset['directed']
    else:
        adj = ori_dataset['undirected']

    features = ori_dataset['features']
    y_train = ori_dataset['train_label']
    y_val = ori_dataset['val_label']
    num_features_nonzero = features[1].shape

    train_mask = ori_dataset['train_mask']
    val_mask = ori_dataset['val_mask']

    dataset = {
        'adj': adj, 'features': features, 
        'label': y_val, 
        'mask': val_mask,
        'num_features_nonzero': num_features_nonzero
    }

    if 'graphsage' == model_name:
        dataset['degrees'] = ori_dataset['degrees']
    elif 'graphsage_meanpool' == model_name:
        dataset['degrees'] = ori_dataset['degrees']
        dataset['row'] = ori_dataset['row']
        dataset['col'] = ori_dataset['col']
    elif 'graphsage_maxpool' == model_name:
        dataset['degrees'] = ori_dataset['degrees']
        dataset['neigh_info'] = ori_dataset['neigh_info']

    return dataset