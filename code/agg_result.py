'''
Aggregate results
Two functions are defined
'''
import os
import pickle as pkl

def agg_result(model_name_list, input_path, input_name_list, outupt_path, output_name):
	'''
	Aggregate results: comma seperated file
	model_name, test_acc, test_acc_std, time, time_std
	The final output is a txt file
	'''
	output_file = open(os.path.join(outupt_path, output_name), 'w')
	output_file.writelines(['model_name, accuracy, accuracy_std, time, time_std'])

	for model_name, each_file in zip(model_name_list, input_name_list):
		input_file = open(os.path.join(input_path, input_name_list), 'rb')
		_, acc, time = pkl.load(input_file)
		output_line = model
		output_line = output_line + ',' + str(acc[0])+ ',' + str(acc[1])
		output_line = output_line + ',' + str(time[0]) + ',' + str(time[1])

		output_file.writelines([output_line])
		input_file.close

	output_file.close()





