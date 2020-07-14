import click
import os
import pandas as pd
import numpy as np

@click.command(name="make_dataset",short_help="create the data structure needed for training")
@click.option('--sequences', '-x', required=True, help='path to the file containing sequence list')
@click.option('--hosts', '-y', required=True, help='path to the file containing corresponding host list ')
@click.option('--outpath', '-o', default='.', help='path where to save the output')
@click.option('--val_split_size', '-v', default=float(0.2), help='select the portion of the data which is used for the validation set')
@click.option('--test_split_size', '-t', default=float(0.2), help='select the portion of the data which is used for the test set')
@click.option('--repeated_undersampling', '-r', is_flag=True, help='generate training files needed for reapeted undersampling while training')


def make_dataset(sequences, hosts, outpath, val_split_size, test_split_size, repeated_undersampling):
	''' create the data structure needed for training

		\b
		Example:
		set input and output parameter
		$ vidhop make_dataset -x /home/user/input/seq.txt -y /home/user/input/host.txt -o /home/user/trainingdata/
		\b
		change the validation set size and provide datastructure for repeated undersampling
		$ vidhop make_dataset -x /home/user/input/seq.txt -y /home/user/input/host.txt -v 0.1 -r
		'''

	if not os.path.isdir(outpath):
		os.makedirs(outpath)

	df_x = pd.read_csv(sequences, names=['sequences'])
	df = pd.read_csv(hosts, names=['hosts'])
	df = df.merge(df_x, left_index=True, right_index=True)
	samples = dict()
	rank = 'hosts'
	hosts = []
	min_samples = 100
	for host in df[rank].unique():
		hosts.append(host)
		df_host = df[df[rank] == host]
		count = df_host[rank].count()
		if count < min_samples:
			print(f"warning number samples for host {host} low, only {count} samples")
		samples.update({host:count})

	number_samples_per_class_to_pick = min(samples.values())

	# create test set
	samples_test = pd.DataFrame()
	for host in hosts:
		df_host = df[df[rank] == host]
		help_test = df_host.sample(n=int(number_samples_per_class_to_pick * test_split_size))
		samples_test = samples_test.append(help_test)

	samples_val_train = df.drop(samples_test.index)
	test_shuffled = samples_test.sample(frac=1)

	X_test = test_shuffled['sequences']
	Y_test = test_shuffled[rank]

	# create validation set
	samples_val = pd.DataFrame()
	for host in hosts:
		df_host = samples_val_train[samples_val_train[rank] == host]
		help_val = df_host.sample(n=int(number_samples_per_class_to_pick * val_split_size))
		samples_val = samples_val.append(help_val)

	# delete val set from val_train set and make random order
	samples_train = samples_val_train.drop(samples_val.index).sample(frac=1)
	samples_val = samples_val.sample(frac=1)

	X_val = samples_val['sequences']
	Y_val = samples_val[rank]


	if repeated_undersampling:
		for i in samples_train[rank].unique():
			directory =  f"{outpath}/train/{i}"
			if not os.path.exists(directory):
				os.makedirs(directory)

		for index, sample in samples_train.iterrows():
			directory = f"{outpath}/train/{sample[rank]}/{index}.csv"
			f = open(directory, 'w')
			f.write(f"{sample.index}\t{sample['sequences']}")
			f.close()


	# nice way with external lib
	# rus = RandomUnderSampler(random_state=0)
	# X_train, Y_train = rus.fit_resample(X_train, Y_train)


	# old way
	number_samples_per_class_to_pick = int(np.ceil(number_samples_per_class_to_pick*(1-val_split_size-test_split_size)))
	samples_train_reduced = pd.DataFrame()
	for host in hosts:
		df_host = samples_train[samples_train[rank] == host]
		help_train = df_host.sample(number_samples_per_class_to_pick)
		samples_train_reduced = samples_train_reduced.append(help_train)

	X_train = samples_train_reduced['sequences']
	Y_train = samples_train_reduced[rank]

	X_train.to_csv(outpath + '/X_train.csv', sep='\t', encoding='utf-8', header=False)
	X_val.to_csv(outpath + '/X_val.csv', sep='\t', encoding='utf-8', header=False)
	X_test.to_csv(outpath + '/X_test.csv', sep='\t', encoding='utf-8', header=False)
	Y_train.to_csv(outpath + '/Y_train.csv', sep='\t', encoding='utf-8', header=False)
	Y_val.to_csv(outpath + '/Y_val.csv', sep='\t', encoding='utf-8', header=False)
	Y_test.to_csv(outpath + '/Y_test.csv', sep='\t', encoding='utf-8', header=False)

if __name__ == '__main__':
	make_dataset()