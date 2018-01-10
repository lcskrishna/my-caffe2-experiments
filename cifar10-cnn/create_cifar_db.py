import pickle
import numpy as np
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

## unpickle the cifar 10 dataset.
def unpickle(file):
	with open(file, "rb") as f:
		dict = pickle.load(f)
	return dict

## creation of caffe2 minidb from two numpy arrays features and labels.
def write_db(db_type, db_name, features, labels):
	db = core.C.create_db(db_type, db_name, core.C.Mode.write)
	transaction = db.new_transaction()
	for i in range(features.shape[0]):
		feature_and_label = caffe2_pb2.TensorProtos()
		feature_and_label.protos.extend([
			utils.NumpyArrayToCaffe2Tensor(features[i]),
			utils.NumpyArrayToCaffe2Tensor(labels[i])
		])
		transaction.put('train_%03d'.format(i), feature_and_label.SerializeToString())
	del transaction
	del db


data1 = unpickle('data_batch_1')
data2 = unpickle('data_batch_2')
data3 = unpickle('data_batch_3')
data4 = unpickle('data_batch_4')
data5 = unpickle('data_batch_5')
test_data = unpickle('test_batch')


X_train1 = data1['data']
Y_train1 = data1['labels']
X_train2 = data2['data']
Y_train2 = data2['labels']
X_train3 = data3['data']
Y_train3 = data3['labels']
X_train4 = data4['data']
Y_train4 = data4['labels']
X_train5 = data5['data']
Y_train5 = data5['labels']
X_test1 = test_data['data']
Y_test1 = test_data['labels']

X_train = np.append(X_train1,X_train2,axis=0)
X_train = np.append(X_train,X_train3,axis=0)
X_train = np.append(X_train,X_train4,axis=0)
Y_train = np.append(Y_train1,Y_train2,axis=0)
Y_train = np.append(Y_train,Y_train3,axis=0)
Y_train = np.append(Y_train,Y_train4,axis=0)

X_validation = np.array(X_train5)
Y_validation = np.array(Y_train5)

X_test = np.array(X_test1)
Y_test = np.array(Y_test1)

print("INFO: The dimensions of the Training Features:")
print(X_train.shape)

print("INFO: The dimensions of the Training Labels:")
print(Y_train.shape)

print("INFO: The dimensions of validation features:")
print(X_validation.shape)

print("INFO: The dimensions of validation labels:")
print (Y_validation.shape)

print("INFO: The dimensions of test features: ")
print(X_test.shape)

print('INFO: The dimensions of test labels : ')
print(Y_test.shape)

'''
#print(X_train.shape[0])
#train_data = X_train.reshape(X_train, (X_train.shape[0], 3, 32, 32))
#print(train_data.shape)

X_train_new = []
for i in range(X_train.shape[0]):
	x = np.array(X_train[i])
	y = x.reshape(3,32,32)
	X_train_new.append(y)

X_train_new = np.array(X_train_new)
print(X_train_new.shape)
'''

## Convert the raw data of images into NCHW format.
def convertToProperFormat(raw):
	raw_float = np.array(raw, dtype = float)/ 255.0
	images = raw_float.reshape([-1, 3, 32, 32])
	#images = images.transpose([0, 2, 3, 1])
	return images

X_train_new = convertToProperFormat(X_train)
print(X_train_new.shape)

X_validation_new = convertToProperFormat(X_validation)
print(X_validation_new.shape)

X_test_new = convertToProperFormat(X_test)
print(X_test_new.shape)


write_db("minidb", "cifar_train.minidb", X_train_new, Y_train)
print ("Train database added.")


write_db("minidb", "cifar_validation.minidb", X_validation_new, Y_validation)
print("Validation database added")

write_db("minidb", "cifar_test.minidb", X_test_new, Y_test)
print("Test database added")

print("RESULT: Successfully created the cifar 10 database")

