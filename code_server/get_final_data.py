import os
import pickle as pkl


data_path = './processed_output'
data_inc_path = './processed_inc_output'

cur_line = ''


write_file_path = os.path.join(data_path, 'global')
write_file_name = 'result'
w = open(os.path.join(write_file_path, write_file_name), 'w+')
w.write('model_name, citeseer, cora, pubmed\n')

for model_name in ['mlp', 'gcn', 'gat', 'dcnn', 'spectralcnn', 'graphsage', 'graphsage_meanpool', 'graphsage_maxpool', 'firstcheb']:
	
	cur_line = model_name

	for dataset_name in ['citeseer', 'cora', 'pubmed']:
		
		if dataset_name == 'pubmed' and model_name == 'dcnn':
			cur_line += ', ' + '(,), (,)'
		else:
			read_path = os.path.join(data_path, dataset_name)
			read_name = model_name + 'rand'

			f = open(os.path.join(read_path, read_name), 'rb')
			_, acc, time = pkl.load(f)

			cur_line = cur_line + ',' + str(acc) + ',' + str(time)

		f.close()

	print(cur_line)
	w.write(cur_line + '\n')

w.close()

