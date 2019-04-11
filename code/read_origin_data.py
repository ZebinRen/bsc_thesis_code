import os
import numpy as np
import pickle
import tensorflow as tf
import scipy.sparse as sp

def parse_id(id_list):
    '''
    This function will take a list of ID
    The ID can be number, string or any object which can be used as the key of dict
    Return value: A dict
        each entry: origin_id, renamed id
        renamed in int, start from 0
    Note: The id is already sorted
    '''
    length = len(id_list)
    rearranged_id = {}
    
    for new_id in range(length):
        rearranged_id[id_list[new_id]] = new_id
    
    return rearranged_id

def dic_to_row_csr(dic):
    '''
    Chnange dict to row_csr
    Note: The key of index: 0 to max
    '''
    row = []
    col = []
    data = []

    for key in dic:
        cur_attr = dic[key]
        for index in range(len(cur_attr)):
            if '1' == cur_attr[index]:
                row.append(key)
                col.append(index)
    data = np.ones(len(row), dtype=np.float64)

    return sp.csr_matrix((data, (row, col)), shape=(len(dic), len(dic[0])), dtype=np.float64)

    
        
def read_origin_cc(path, citename, contentname, cate_map):
    '''
    Read the origin file for citeseer and core
    They have the same file structure
    '''
    #Open file
    attr_reader = open(os.path.join(path, contentname), 'r')
    graph_reader = open(os.path.join(path, citename), 'r')

    #attr: origin_node_id : attr list
    attr = {}
    #cate: origin_node_id : remapped_cate
    cate = {}
    #a list of id
    id_list = []

    #read id, arrtibute and cate
    for line in attr_reader:
        line = line.strip()
        line = line.split('\t')
        id_list.append(line[0])
        attr[line[0]] = line[1 : len(attr) - 1]
        cate[line[0]] = cate_map[line[-1]]
    
    #rename id
    #id_map is a dict old_id -> new_id
    id_map = parse_id(id_list)

    #Change the id in attr and cate
    new_attr = {}
    new_cate = {}
    length = len(id_list)

    for old_id in id_list:
        new_attr[id_map[old_id]] = attr[old_id]
        new_cate[id_map[old_id]] = cate[old_id]
    
    attr = new_attr
    cate = new_cate

    #Read cite data, create sparse matrix
    undir_row = []
    undir_col = []
    undir_data = []
    dir_row = []
    dir_col = []
    dir_data = []
    
    
    for line in graph_reader:
        line = line.strip()
        line = line.split('\t')
        
        if not ((line[0] in id_map) and (line[1] in id_map)):
            continue
        
        entry_row = id_map[line[0]]
        entry_col = id_map[line[1]]
        
        dir_row.append(entry_row)
        dir_col.append(entry_col)

        undir_row.append(entry_row)
        undir_col.append(entry_col)
        undir_row.append(entry_col)
        undir_col.append(entry_row)

        dir_data = np.ones(len(dir_row), dtype=np.float64)
        undir_data = np.ones(len(undir_row), dtype=np.float64)

    directed = sp.csr_matrix((dir_data, (dir_row, dir_col)), shape=(length, length), dtype=np.float64)
    undirected = sp.csr_matrix((undir_data, (undir_row, undir_col)), shape=(length, length), dtype=np.float64 )

    attr_reader.close()
    graph_reader.close()
    attr = dic_to_row_csr(attr)

    return directed, undirected, attr, cate



def read_origin_pubmed(path, directedname, nodename):
    graph_reader = open(os.path.join(path, directedname), 'r')
    node_reader = open(os.path.join(path, nodename), 'r')


    #The first and second lines are rubbish
    graph_reader.readline()
    graph_reader.readline()

    #The first line is rubbish, the second line is attributs' name
    node_reader.readline()
    #handle the attribute name
    attr_name = node_reader.readline()
    attr_name = attr_name.strip()
    attr_name = attr_name.split('\t')
    attr_name = attr_name[1: -1]
    
    for i in range(len(attr_name)):
        tmp = attr_name[i]
        tmp = tmp.split(':')
        attr_name[i] = tmp[1]
    
    #Now, attribute list is a list of the name of attributes

    #Each node doesn't has all the attributes, so assign each attribute with a index
    attr_index = {}
    for i in range(len(attr_name)):
        attr_index[attr_name[i]] = i
    
    #Now, read attributes
    #Structure id, cate, attributes_list, summary
    attr_list = []
    id_list = []
    cate = {}
    for line in node_reader:
        line = line.strip()
        line = line.split('\t')
        
        id = line[0]
        cate[id] = int((line[1].split('='))[1])
        id_list.append(id)

        attr_list.append(line)

    #Sort and rename id for converence
    node_number = len(id_list)
    new_id = parse_id(id_list)
    length = len(id_list)

    #change the id-cate map to new id
    new_cate = {}
    for id in id_list:
        new_cate[new_id[id]] = cate[id]
    cate = new_cate

    content_row = []
    content_col = []
    content_data = []
    content_row_len = len(attr_list)
    content_col_len = len(attr_index)

    #Build attribute matrix(CSR)
    for line in attr_list:
        attr = line[2: -1]
        row = new_id[line[0]]
        for element in attr:
            element = element.split('=')
            col = attr_index[element[0]]
            content_row.append(row)
            content_col.append(col)
            content_data.append(float(element[1]))
    content = sp.csr_matrix((content_data, (content_row, content_col)), 
            shape=(content_row_len, content_col_len), dtype=np.float64)

    #Read cite data, create sparse matrix
    undir_row = []
    undir_col = []
    undir_data = []
    dir_row = []
    dir_col = []
    dir_data = []

    for line in graph_reader:
        line = line.strip()
        line = line.split('\t')
        #TODO: Check out the meaning of the first element and finihs this part

        #I don't know line[0] standa for what
        tag = line[0]
        row_id = line[1]
        col_id = line[3]
        row_id = new_id[(row_id.split(':'))[1]]
        col_id = new_id[(col_id.split(':'))[1]]

        #Create index and data list
        #If tag means the weight, change 1 to tag
        dir_row.append(row_id)
        dir_col.append(col_id)
        dir_data.append(1)
        undir_row.append(row_id)
        undir_col.append(col_id)
        undir_row.append(col_id)
        undir_col.append(row_id)
        undir_data.append(1)
        undir_data.append(1)
    
    directed = sp.csr_matrix((dir_data, (dir_row, dir_col)), shape=(length, length), dtype=np.float64)
    undirected = sp.csr_matrix((undir_data, (undir_row, undir_col)), shape=(length, length), dtype=np.float64 )
    

    return directed, undirected, content, cate

    
def run_main():
    #Path to read files
    citeseer_path = '../origin_data/citeseer'
    citeseer_cite_name = 'citeseer.cites'
    citeseer_content_name = 'citeseer.content'
    cora_path =  '../origin_data/cora'
    cora_cite_name = 'cora.cites'
    cora_content_name = 'cora.content'
    pubmed_path = '../origin_data/pubmed'
    pubmed_cite_name = 'Pubmed-Diabetes.DIRECTED.cites.tab'
    pubmed_content_name = 'Pubmed-Diabetes.NODE.paper.tab'

    #Path to save files
    citeseer_save_path = '../processed_data'
    citeseer_directed = 'citeseer_directed'
    citeseer_undirected = 'citeseer_undirected'
    citeseer_attribute = 'citeseer_attribute'
    citeseer_cate = 'citeseer_cate'
    cora_save_path =  '../processed_data'
    cora_directed = 'cora_directed'
    cora_undirected = 'cora_undirected'
    cora_attribute = 'cora_attirbute'
    cora_cate = 'cora_cate'
    pubmed_save_path =  '../processed_data'
    pubmed_directed = 'pubmed_directed'
    pubmed_undirected = 'pubmed_undirected'
    pubmed_attribute = 'pubmed_attribute'
    pubmed_cate ='pubmed_cate'

    #Open_write_file
    citeseer_directed_w = open(os.path.join(citeseer_save_path, citeseer_directed), 'wb')
    citeseer_undirected_w = open(os.path.join(citeseer_save_path, citeseer_undirected), 'wb')
    citeseer_attribute_w = open(os.path.join(citeseer_save_path, citeseer_attribute), 'wb')
    citeseer_cate_w = open(os.path.join(citeseer_save_path, citeseer_cate), 'wb')

    cora_directed_w = open(os.path.join(cora_save_path, cora_directed), 'wb')
    cora_undirected_w = open(os.path.join(cora_save_path, cora_undirected), 'wb')
    cora_attribute_w = open(os.path.join(cora_save_path, cora_attribute), 'wb')
    cora_cate_w = open(os.path.join(cora_save_path, cora_cate), 'wb')

    pubmed_directed_w = open(os.path.join(pubmed_save_path, pubmed_directed), 'wb')
    pubmed_undirected_w = open(os.path.join(pubmed_save_path, pubmed_undirected), 'wb')
    pubmed_attribute_w = open(os.path.join(pubmed_save_path, pubmed_attribute), 'wb')
    pubmed_cate_w = open(os.path.join(pubmed_save_path, pubmed_cate), 'wb')
    

    cate_map_citeseer= {
        'Agents':0, 'AI': 1, 'DB': 2, 'IR': 3, 'ML': 4, 'HCI': 5
    } 

    cate_map_cora = {
        'Case_Based': 0, 'Genetic_Algorithms': 1, 'Neural_Networks': 2, 'Probabilistic_Methods': 3, 
        'Reinforcement_Learning': 4, 'Rule_Learning': 5, 'Theory': 6
    }

    #read citeseer
    cite_d, cite_un, cite_attr, cite_cate = read_origin_cc(citeseer_path, citeseer_cite_name,
            citeseer_content_name, cate_map_citeseer)

    
    #read cora
    cora_d, cora_un, cora_attr, cora_cate = read_origin_cc(cora_path, cora_cite_name, 
            cora_content_name, cate_map_cora)

    #read pubmed
    pubmed_d, pubmed_un, pubmed_attr, pubmend_cate = read_origin_pubmed(pubmed_path, 
            pubmed_cite_name, pubmed_content_name)

    #Save to file
    pickle.dump(cite_d, citeseer_directed_w)
    pickle.dump(cite_un, citeseer_undirected_w)
    pickle.dump(cite_attr, citeseer_attribute_w)
    pickle.dump(cite_cate, citeseer_cate_w)

    
    pickle.dump(cora_d, cora_directed_w)
    pickle.dump(cora_un, cora_undirected_w)
    pickle.dump(cora_attr, cora_attribute_w)
    pickle.dump(cora_cate, cora_cate_w)

    pickle.dump(pubmed_d, pubmed_directed_w)
    pickle.dump(pubmed_un, pubmed_undirected_w)
    pickle.dump(pubmed_attr, pubmed_attribute_w)
    pickle.dump(pubmed_cate, pubmed_cate_w)

    citeseer_directed_w.close()
    citeseer_undirected_w.close()
    citeseer_attribute_w.close()
    citeseer_cate_w.close()


    cora_directed_w.close()
    cora_undirected_w.close()
    cora_attribute_w.close()
    cora_cate_w.close()

    pubmed_directed_w.close()
    pubmed_undirected_w.close()
    pubmed_attribute_w.close()
    pubmed_cate_w.close()

    return cite_d, cite_un, cite_attr, cite_cate 

if __name__ == '__main__':
    run_main()