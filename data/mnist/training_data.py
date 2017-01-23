import os
import numpy as np
import struct
import exceptions

TRAINING_FILES = {	'images':'train-images-idx3-ubyte',
					'labels':'train-labels-idx1-ubyte' }

TEST_FILES = 	 {	'images':'t10k-images-idx3-ubyte',
					'labels':'t10k-labels-idx1-ubyte' }

NUM_LABELS = 10

def load_labels(filename):
	
	# from http://yann.lecun.com/exdb/mnist/

	# [offset] [type]          [value]          [description] 
	# 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
	# 0004     32 bit integer  60000            number of items 
	# 0008     unsigned byte   ??               label 
	# 0009     unsigned byte   ??               label 
	# ........ 
	# xxxx     unsigned byte   ??               label
	# The labels values are 0 to 9.

	with open( filename, 'rb' ) as binary_data:
		magic, num_items = struct.unpack_from( '>ii', binary_data.read(8) )
		
		if magic != 2049:
			raise exceptions.IOError('Unrecognised MNIST label file')

		labels = np.fromfile(binary_data, dtype=np.ubyte)

		# Encode labels as one-hot vectors
		one_hot = np.zeros( [num_items, NUM_LABELS] )
		one_hot[ np.arange(num_items), labels ] = 1.0

		return one_hot


def load_images(filename):

	# [offset] [type]          [value]          [description] 
	# 0000     32 bit integer  0x00000803(2051) magic number 
	# 0004     32 bit integer  60000            number of images 
	# 0008     32 bit integer  28               number of rows 
	# 0012     32 bit integer  28               number of columns 
	# 0016     unsigned byte   ??               pixel 
	# 0017     unsigned byte   ??               pixel 
	# ........ 
	# xxxx     unsigned byte   ??               pixel

	with open( filename, 'rb' ) as binary_data:

		magic, num_images, num_rows, num_cols = struct.unpack_from( '>iiii', binary_data.read(16) )

		if magic != 2051:
			raise exceptions.IOError('Unrecognised MNIST image file')

		images_bytes = np.fromfile(binary_data, dtype=np.ubyte)

		# Assemble into a large matrix, one row per image. 0.0 is background, 1.0 is foreground
		images = np.vstack( np.split(images_bytes, num_images)  ) / 255.0
		
	return (num_rows, num_cols, images)


def load():

	download_dir = os.path.abspath( os.path.join( os.path.dirname(__file__), 'download' ) )

	tr_rows, tr_cols, training_images = load_images( os.path.join( download_dir, TRAINING_FILES['images']) )
	training_labels = load_labels( os.path.join( download_dir, TRAINING_FILES['labels']) )

	tst_rows, tst_cols, test_images = load_images( os.path.join( download_dir, TEST_FILES['images']) )
	test_labels = load_labels( os.path.join( download_dir, TEST_FILES['labels']) )

	if tr_rows != tst_rows or tr_cols != tst_cols:
		raise exceptions.RuntimeError('Training and test image sets have different dimensions')

	return {	'training': {'images': training_images,
						 	 'labels': training_labels},

				'test':     {'images': test_images,
						 	 'labels': test_labels },

				'rows_cols': [tr_rows, tr_cols]
			}


