## DB Manager is a utility program for caffe2 that can be used for creating a db (minidb format).
## @Author: Chaitanya Sri Krishna Lolla

import numpy as np
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

'''
This method should be used for converting a numpy arrays into various databases types for caffe2
Note : Make sure to pass the features and labels in numpy arrays and features in the format of NCHW for better consistency.
@arg : db_type - Database type (minidb, lmdb etc)
@arg : db_name - name of the database that is intended to be saved.
@arg : features - numpy arrays of features (assumption of the numpy array shape should be in NCHW order)
@arg : labels - numpy arrays of labels.

'''
def write_db(db_type, db_name, features, labels):
	db = core.C.create_db(db_type, db_name, core.C.Mode.write)
	transaction = db.new_transaction()
	
	for i in range(features.shape[0]):
		feature_and_label = caffe2_pb2.TensorProtos()
		feature_and_label.protos.extend([
			utils.NumpyArrayToCaffe2Tensor(features[i]),
			utils.NumpyArrayToCaffe2Tensor(labels[i])
		])
		transaction.put('train_%03d'.format(i),	feature_and_label.SerializeToString())

	del transaction
	del db
	print("INFO: Successfully created the database")

'''
This method takes a raw data of the images and converts into NCHW format.
@arg: raw - raw image data (flattened)
@arg: num_channels - number of channels of the images.
@arg: width - width of the image.
@arg: height - height of the image.

'''
def convertRawDataToNCHWFormat(raw, num_channels, width, height):
	raw_float = np.array(raw, dtype = float)/255.0
	images = raw_float.reshape([-1, num_channels, width, height])
	print("INFO: Conversion successful")
	return images
