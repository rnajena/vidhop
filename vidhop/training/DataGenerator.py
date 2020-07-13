import numpy as np
import tensorflow as tf
import os
import multiprocessing.pool
from functools import partial
from random import sample as randsomsample
import pandas as pd
import sys
sys.path.insert(0, '/home/go96bix/projects/Masterarbeit/ML')
from vidhop.DataParsing import DataParsing_main

import warnings

class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'

	def __init__(self, directory, classes=None, number_subsequences=32, dim=(32, 32, 32), n_channels=6,
				 n_classes=10, shuffle=True, n_samples=None, seed=None, faster=True, online_training=False, repeat=True,
				 use_spacer=False, randomrepeat=False, sequence_length=50, number_samples_per_batch=32 , **kwargs):
		'Initialization'
		self.directory = directory
		self.classes = classes
		self.dim = dim
		self.labels = None
		self.list_IDs = None
		self.n_channels = n_channels
		self.shuffle = shuffle
		self.seed = seed
		self.online_training = online_training
		self.repeat = repeat
		self.use_spacer = use_spacer
		self.randomrepeat = randomrepeat
		self.maxLen = kwargs.get("maxLen", None)
		self.sequence_length = sequence_length

		if number_subsequences == 1:
			self.shrink_timesteps = False
		else:
			self.shrink_timesteps = True

		self.number_subsequences = number_subsequences

		if faster == True:
			self.faster = 16
		elif type(faster) == int and faster > 0:
			self.faster = faster
		else:
			self.faster = 1

		self.number_samples_per_batch = number_samples_per_batch * self.faster

		self.number_samples_per_class_to_pick = n_samples

		if not classes:
			classes = []
			for subdir in sorted(os.listdir(directory)):
				if os.path.isdir(os.path.join(directory, subdir)):
					classes.append(subdir)
			self.classes = classes

		self.n_classes = len(classes)
		self.class_indices = dict(zip(classes, range(len(classes))))

		# want a dict which contains dirs and number usable files
		pool = multiprocessing.pool.ThreadPool()
		function_partial = partial(_count_valid_files_in_directory,
								   white_list_formats={'csv'},
								   follow_links=None,
								   split=None)
		self.samples = pool.map(function_partial, (os.path.join(directory, subdir) for subdir in classes))
		self.samples = dict(zip(classes, self.samples))

		results = []

		for dirpath in (os.path.join(directory, subdir) for subdir in classes):
			results.append(pool.apply_async(_list_valid_filenames_in_directory,
											(dirpath, {'csv'}, None, self.class_indices, None)))

		self.filename_dict = {}
		for res in results:
			classes, filenames = res.get()
			for index, class_i in enumerate(classes):
				self.filename_dict.update({f"{class_i}_{index}": filenames[index]})

		pool.close()
		pool.join()

		if not n_samples:
			self.number_samples_per_class_to_pick = min(self.samples.values())

		# self.elmo_embedder = Elmo_embedder()
		self.elmo_embedder = None

		self.on_epoch_end()

	# in images wird ein groesses arr classes gemacht (fuer alle sampels) darin stehen OHE die Class
	# erstelle filename liste in der die zugehoerige file adresse steht
	# laesst sich mergen mit version die oben verlinked

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.number_samples_per_batch))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index * self.number_samples_per_batch:(index + 1) * self.number_samples_per_batch]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y, sample_weight = self.__data_generation(list_IDs_temp, indexes)

		return (X, y)

	def on_epoch_end(self):
		'make X-train sample list'
		"""
		1. go over each class
		2. select randomly #n_sample samples of each class
		3. add selection list to dict with class as key 
		"""

		self.class_selection_path = np.array([])
		self.labels = np.array([])
		for class_i in self.classes:
			samples_class_i = randsomsample(range(0, self.samples[class_i]), self.number_samples_per_class_to_pick)
			self.class_selection_path = np.append(self.class_selection_path,
												  [self.filename_dict[f"{self.class_indices[class_i]}_{i}"] for i in
												   samples_class_i])
			self.labels = np.append(self.labels, [self.class_indices[class_i] for i in samples_class_i])

		self.list_IDs = self.class_selection_path

		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			if self.seed:
				np.random.seed(self.seed)
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp, indexes):
		pool = multiprocessing.pool.ThreadPool()
		'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
		# Initialization
		# X = np.empty((self.batch_size, self.dim, self.n_channels),dtype='str')
		X = np.empty((self.number_samples_per_batch), dtype=object)
		Y = np.empty((self.number_samples_per_batch), dtype=int)

		sample_weight = np.array([])

		def load_csv(sample):
			X_i = pd.read_csv(os.path.join(self.directory, sample), delimiter='\t', dtype='str', header=None)[1].values[0]
			return X_i

		# Generate data
		samples = pool.map(load_csv,list_IDs_temp)
		X = np.array(samples)
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			# load tsv, parse to numpy array, get str and set as value in X[i]
			# X[i] = pd.read_csv(os.path.join(self.directory, ID), delimiter='\t', dtype='str', header=None)[1].values[0]
			# sample_weight = np.append(sample_weight, 1)
			# if len(X[i]) < self.dim:
			# 	X[i] = "-" * self.dim
			# 	sample_weight[i] = 0

			# Store class
			Y[i] = self.labels[indexes[i]]

		sample_weight = np.array([[i] * self.number_subsequences for i in sample_weight]).flatten()
		if self.maxLen == None:
			maxLen = self.number_subsequences * self.dim
		else:
			maxLen = self.maxLen

		# original_length = 50
		# start_float = (original_length - self.sequence_length) / 2
		# start = math.floor(start_float)
		# stop = original_length - math.ceil(start_float)

		# # amino = "GALMFWKQESPVICYHRNDTU"
		# amino = "GALMFWKQESPVICYHRNDTUOBZX"
		# encoder = LabelEncoder()
		# encoder.fit(list(amino))
		# X = parse_amino(x=[[i[start:stop]] for i in X], encoder=encoder)

		# X = self.elmo_embedder.elmo_embedding(X, start, stop)
		#
		# X = seqvec.embed_sentence([i[start:stop] for i in X])
		def encode_sample(sample):
			X_i = DataParsing_main.encode_string(maxLen=maxLen, x=str(sample), repeat=self.repeat, use_spacer=self.use_spacer)
			return X_i

		X_wrong_shape = np.array(pool.map(encode_sample,X))
		X = np.array(X_wrong_shape).reshape((X_wrong_shape.shape[0],-1,6))
		# X = DataParsing.encode_string(maxLen=maxLen, x=X, repeat=self.repeat, use_spacer=self.use_spacer)
		# assert self.shrink_timesteps != True or self.online_training != True, "online_training shrinks automatically " \
		#                                                                       "the files, please deactivate shrink_timesteps"

		if self.online_training:
			X, Y = DataParsing_main.manipulate_training_data(X=X, Y=Y, subSeqLength=self.dim,
															 number_subsequences=self.number_subsequences)
		elif self.shrink_timesteps:
			X, Y, batchsize = DataParsing_main.shrink_timesteps(input_subSeqlength=self.dim, X=X, Y=Y)

		pool.close()
		pool.join()
		return X, tf.keras.utils.to_categorical(Y, num_classes=self.n_classes), sample_weight


def _count_valid_files_in_directory(directory, white_list_formats, split,
									follow_links):
	"""
	Copy from keras 2.1.5
	Count files with extension in `white_list_formats` contained in directory.

	Arguments:
		directory: absolute path to the directory
			containing files to be counted
		white_list_formats: set of strings containing allowed extensions for
			the files to be counted.
		split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
			account a certain fraction of files in each directory.
			E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
			of images in each directory.
		follow_links: boolean.

	Returns:
		the count of files with extension in `white_list_formats` contained in
		the directory.
	"""
	num_files = len(
		list(_iter_valid_files(directory, white_list_formats, follow_links)))
	if split:
		start, stop = int(split[0] * num_files), int(split[1] * num_files)
	else:
		start, stop = 0, num_files
	return stop - start


def parse_amino(x, encoder):
	out = []
	for i in x:
		# dnaSeq = i[1].upper()
		dnaSeq = i[0].upper()
		encoded_X = encoder.transform(list(dnaSeq))
		out.append(encoded_X)
	return np.array(out)


def _list_valid_filenames_in_directory(directory, white_list_formats, split,
									   class_indices, follow_links):
	"""Lists paths of files in `subdir` with extensions in `white_list_formats`.
	Copy from keras-preprocessing 1.0.9
	# Arguments
		directory: absolute path to a directory containing the files to list.
			The directory name is used as class label
			and must be a key of `class_indices`.
		white_list_formats: set of strings containing allowed extensions for
			the files to be counted.
		split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
			account a certain fraction of files in each directory.
			E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
			of images in each directory.
		class_indices: dictionary mapping a class name to its index.
		follow_links: boolean.

	# Returns
		 classes: a list of class indices
		 filenames: the path of valid files in `directory`, relative from
			 `directory`'s parent (e.g., if `directory` is "dataset/class1",
			the filenames will be
			`["class1/file1.jpg", "class1/file2.jpg", ...]`).
	"""
	dirname = os.path.basename(directory)
	if split:
		num_files = len(list(
			_iter_valid_files(directory, white_list_formats, follow_links)))
		start, stop = int(split[0] * num_files), int(split[1] * num_files)
		valid_files = list(
			_iter_valid_files(
				directory, white_list_formats, follow_links))[start: stop]
	else:
		valid_files = _iter_valid_files(
			directory, white_list_formats, follow_links)
	classes = []
	filenames = []
	for root, fname in valid_files:
		classes.append(class_indices[dirname])
		absolute_path = os.path.join(root, fname)
		relative_path = os.path.join(
			dirname, os.path.relpath(absolute_path, directory))
		filenames.append(relative_path)

	return classes, filenames


def _iter_valid_files(directory, white_list_formats, follow_links):
	"""Iterates on files with extension in `white_list_formats` contained in `directory`.

	# Arguments
		directory: Absolute path to the directory
			containing files to be counted
		white_list_formats: Set of strings containing allowed extensions for
			the files to be counted.
		follow_links: Boolean.

	# Yields
		Tuple of (root, filename) with extension in `white_list_formats`.
	"""

	def _recursive_list(subpath):
		return sorted(os.walk(subpath, followlinks=follow_links),
					  key=lambda x: x[0])

	for root, _, files in _recursive_list(directory):
		for fname in sorted(files):
			if fname.lower().endswith('.tiff'):
				warnings.warn('Using ".tiff" files with multiple bands '
							  'will cause distortion. Please verify your output.')
			if get_extension(fname) in white_list_formats:
				yield root, fname


def get_extension(filename):
	"""Get extension of the filename

	There are newer methods to achieve this but this method is backwards compatible.
	"""
	return os.path.splitext(filename)[1].strip('.').lower()
