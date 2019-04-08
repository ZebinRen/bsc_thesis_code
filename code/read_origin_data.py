import os
import numpy as np

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
        
def read_origin_cc(path, citename, contentname, cate_map):
    '''
    Read the origin file for citeseer and core
    They have the same file structure
    '''
    #Open file
    attr_reader = open(os.path.join(path, contentname), 'r')
    graph_reader = open(os.path.join(path, citename), 'r')

    attr = {}
    cate = {}
    id_list = []

    #read id, arrtibute and cate
    for line in attr_reader:
        line = line.strip()
        line = line.split('\t')
        id_list.append(line[0])
        attr[line[0]] = line[1 : len(attr) - 2]
        cate[line[0]] = cate_map[line[-1]]
    
    #reorder id, rename id
    #id_map is a dict old_id -> new_id
    id_list.sort()
    id_map = parse_id(id_list)

    '''
    #DEBUG:
    w_cate = open('cate.txt', 'w')
    w_attr = open('attr.txt', 'w')
    w_id_list = open('id_list.txt', 'w')
    
    for t in cate:
        w_cate.write(t + '\n')
    for i in id_list:
        w_id_list.write(i + '\n')
    w_cate.close()
    w_attr.close()
    w_id_list.close()
    #DEBUG FINSIHED
    '''

    #Change the id in attr and cate
    new_attr = {}
    new_cate = []
    length = len(id_list)

    for old_id in id_list:
        new_attr[id_map[old_id]] = attr[old_id]
        new_cate.append(cate[old_id])
    
    attr = new_attr
    cate = new_cate

    #Read cite data
    directed = np.zeros((length, length))
    undirected = np.zeros((length, length))

    cite_cate = []
    for line in graph_reader:
        line = line.strip()
        line = line.split('\t')
        
        cite_cate.append(line[0])
        cite_cate.append(line[1])
        
        '''
        entry_row = id_map[line[0]]
        entry_col = id_map[line[1]]

        directed[entry_row][entry_col] = 1
        undirected[entry_row][entry_col] = 1
        undirected[entry_col][entry_row] = 1
        '''
    '''
    cite_cate_set = set(cite_cate)
    cite_cate = list(cite_cate_set)
    
    print(len(id_list))
    print(len(cite_cate_set))
    print(len(cite_cate))
    '''
    attr_reader.close()
    graph_reader.close()


    return cite_directed, cite_undirected, attr, cate
'''
def read_origin_cora(path, citename, attrname):
    content = open(os.path.join(path, contentname), 'r')
    cite = open(os.path.join(path, citename), 'r')


    content = {}
    tag = {}
    cite = []
'''



def read_origin_pubmed(path, directedname, nodename):
    graph_reader = open(os.path.join(path, directedcitesname), 'r')
    node_reader = open(os.path.join(path, nodepapername), 'r')


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
    attr_list[]
    id_list[]
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
    new_id = parse_id(id_list.sort())

    #change the id-cate map to new id
    new_cate = []
    for id in id_list:
        new_cate.append(cate[id])
    cate = new_cate


    content = numpy.zeros((node_number, 500))

    #Build attribute matrix
    for line in attr_list:
        attr = line[2: -1]
        row = new_id(line[0])
        for element in line:
            element = element.split('=')
            col = attr_index[element[0]]
            content[row][col] = float(element[1])
    
    directed = numpy.zeros(node_number, node_number)
    undirected = numpy.zeros(node_number, node_number)

    for line in graph_reader:
        line.strip()
        line = line.split('\t')
        #TODO: Check out the meaning of the first element and finihs this part

    

    return directed. undirected, content, cate

    
if __name__ == '__main__':
    tag_map_citeseer= {
        'Agents':0, 'AI': 1, 'DB': 2, 'IR': 3, 'ML': 4, 'HCI': 5
    } 

    tag_map_cora = {
        'Case_Based': 0, 'Genetic_Algorithms': 1, 'Neural_Networks': 2, 'Probabilistic_Methods': 3, 
        'Reinforcement_Learning': 4, 'Rule_Learning': 5, 'Theory': 6
    }

    #read citeseer
    cite_d, cite_und, cite_content, cite_tag = read_origin_cc('../origin_data/citeseer', 'citeseer.cites',
            'citeseer.content', tag_map_citeseer)

    #read cora
    cora_d, cora_und, cora_content, cora_tag = read_origin_cc('../origin_data/cora', 'cora.cites', 
            'cora.content', tag_map_cora)

    #read pubmed