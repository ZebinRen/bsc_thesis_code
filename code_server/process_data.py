import pickle as pkl
import numpy as np

def pad_list(ori_list, pad_number, length):
    '''
    pad list to a given length
    '''
    if len(ori_list) > length:
        raise 'origin list is longer than padded length'
    if len(ori_list) == length:
        return ori_list

    pad = [pad_number] * (length - len(ori_list))

    return ori_list + pad

def pad_res_list(res_list, length):
    '''
    pad each item in the res_list to a given length
    '''

    res = []

    for item in res_list:
        tmp_res = pad_list(item, item[-1], length)
        res.append(tmp_res)

    return res


def process_output(train_info_list, acc_list, time_list):
    '''
    process the output data
    parameters:
    train_info_list: contains train info
    acc_list: list of accuacy
    time_list: list of time used
    Data_strucurs:
    train_info_data: list of list
        outter_list: each instance of dataset
        inner_list: multiple evaluation for the same dataset
        each element in the inner list: dict 
        keys: train_loss, train_acc, val_loss, val_acc
    acc_list: list of list
        outter_list: each instances of dataset
        inner_list: multiple evaluation for the samge dataset
    time_list: same structure as acc_list
    output:
    train_info: A single dictioary, keys are same as origin
    accuracy: (ave_accuracy, std)
    time: (ave_time, std)
    '''

    #initialize parameters
    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []
    max_tloss = 0
    max_tacc = 0
    max_vloss = 0
    max_vacc = 0

    for dataset_train_info in train_info_list:
        for item in dataset_train_info:
            t_loss.append(item['train_loss'])
            t_acc.append(item['train_acc'])
            v_loss.append(item['val_loss'])
            v_acc.append(item['val_acc'])
            if max_tloss < len(t_loss[-1]):
                max_tloss = len(t_loss[-1])
            if max_tacc < len(t_acc[-1]):
                max_tacc = len(t_acc[-1])
            if max_vloss < len(v_loss[-1]):
                max_vloss = len(v_loss[-1])
            if max_vacc < len(v_acc[-1]):
                max_vacc = len(v_acc[-1])

    #pad list 
    t_loss = pad_res_list(t_loss, max_tloss)
    t_acc = pad_res_list(t_acc, max_tacc)
    v_loss = pad_res_list(v_loss, max_vloss)
    v_acc = pad_res_list(v_acc, max_vacc)

    #take average
    #print(t_loss)
    t_loss_ave = np.average(t_loss, axis = 0)
    t_acc_ave = np.average(t_acc, axis=0)
    v_loss_ave = np.average(v_loss, axis=0)
    v_acc_ave = np.average(v_acc, axis=0)

    #Return dictionary
    train_info = {
        'train_loss': t_loss_ave,
        'train_acc' : t_acc_ave,
        'val_loss' : v_loss_ave,
        'val_acc' : v_acc_ave
    }



    #compute accuracy and time 
    eva_time = len(acc_list[0])

    #add each sub_list 
    acc_list = np.sum(acc_list, axis = 1)
    time_list = np.sum(time_list, axis = 1)

    #Take average
    acc_list = acc_list/eva_time
    time_list = time_list/eva_time

    #compute ave
    acc_ave = np.average(acc_list)
    time_ave = np.average(time_list)

    #compute std
    acc_std = np.std(acc_list)
    time_std = np.std(time_list)


    return train_info, [acc_ave, acc_std], [time_ave, time_std]


