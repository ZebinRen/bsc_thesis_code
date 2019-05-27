import os
import pickle as pkl


data_path = './processed_output'
data_inc_path = './processed_inc_output'

cur_line = ''

for model_name in ['mlp', 'gcn', 'gat', 'dcnn', 'spectralcnn', 'graphsage', 'graphsagemeanpool', 'graphsagemaxpool', 'firstcheb']:
	write_file_path = os.path.join(read_path, 'global')
	write_file_name = 'result'
	w = open(os.path.join(write_file_path, write_file_name), 'w')
	w.write_lines(['model_name, citeseer, cora, pubmed'])
	for dataset_name in ['citeseer', 'cora', 'pubmed']:
		cur_line = dataset_name

		read_path = os.path.join(data_path, dataset_name)
		read_name = model_name + 'rand'

		f = open(os.path.join(read_path, read_name), 'rb')
		_, acc, time = pkl.load(f)

		cur_line = cur_line+ ',' + str(acc) + str(time)

		f.close()

	w.write_lines['cur_line']
w.close()

