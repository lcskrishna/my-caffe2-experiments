## Author: Chaitanya Sri Krishna Lolla
## Caffe2 CIFAR 10 dataset.

from caffe2.python import core, model_helper, workspace, brew
import numpy as np
import os


data_path = os.getcwd()
output_path = os.path.join(data_path, 'output')

if os.path.exists(output_path):
	print("INFO: Output path already present.")
else:
	print("WARN: Output path not present. hence creating a folder")
	os.makedirs(output_path)

core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

## Input layer.
def AddInput(model, batch_size, db, db_type):
	#db_reader = model.CreateDB([], "db_reader", db = db, db_type = db_type)
	#model.TensorsProtosDBInput([db_reader], ["data", "label"],batch_size = 16)
	data_uint8, label = model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size = batch_size, db = db, db_type = db_type)
	data = model.Cast(data_uint8, "data", to= core.DataType.FLOAT)
	#data = model.Scale(data, data, scale = float(1./256))
	data = model.StopGradient(data, data)
	return data, label

def AddNetModel(model, data):
	conv1 = brew.conv(model, data, 'conv1', dim_in = 3, dim_out = 32, kernel = 5, pad = 2)
	pool1 = brew.max_pool(model, conv1, 'pool1', kernel = 3, stride = 2)
	relu1 = brew.relu(model, pool1, 'relu1')
	conv2 = brew.conv(model, relu1, 'conv2', dim_in = 32, dim_out = 32, kernel = 5, pad = 2)
	relu2 = brew.relu(model, conv2, 'relu2')
	pool2 = brew.max_pool(model, relu2, 'pool2', kernel = 3, stride = 2)
	conv3 = brew.conv(model, pool2, 'conv3', dim_in = 32, dim_out = 64, kernel = 5, pad = 2)
	relu3 = brew.relu(model, conv3, 'relu3')
	pool3 = brew.max_pool(model, relu3, 'pool3', kernel = 3, stride = 2)
	fc1 = brew.fc(model, pool3, 'fc1', dim_in = 3 * 3  * 64 , dim_out = 64)
	fc2 = brew.fc(model, fc1, 'fc2', dim_in = 64, dim_out = 10)
	softmax = brew.softmax(model, fc2, 'softmax')
	return softmax

def AddAccuracy(model, softmax, label):
	accuracy = brew.accuracy(model, [softmax, label], "accuracy")
	return accuracy

def AddTrainingParameters(model, softmax, label):
	xent = model.LabelCrossEntropy([softmax, label], 'xent')
	loss = model.AveragedLoss(xent, "loss")
	AddAccuracy(model, softmax, label)
	model.AddGradientOperators([loss])
	ITER = brew.iter(model, "iter")
	LR = model.LearningRate(ITER, "LR", base_lr = -0.1, policy = "step", stepsize = 1, gamma = 0.999)
	ONE = model.param_init_net.ConstantFill([], "ONE", shape = [1], value = 1.0)
	
	for param in model.params:
		param_grad = model.param_to_grad[param]
		model.WeightedSum([param, ONE, param_grad, LR], param)

def AddBookKeepingOperators(model):
	model.Print('accuracy', [], to_file = 1)
	model.Print('loss', [], to_file = 1)
	
	for param in model.params:
		model.Summarize(param, [], to_file = 1)
		model.Summarize(model.param_to_grad[param], [], to_file = 1)

### Model and parameters creation.
arg_scope = {"order" : "NCHW"}
train_model = model_helper.ModelHelper(name = "cifar10_train", arg_scope = arg_scope)
data, label = AddInput(train_model, batch_size = 1000, db = os.path.join(data_path, 'cifar_train.minidb'), db_type = "minidb")
softmax = AddNetModel(train_model, data)
AddTrainingParameters(train_model, softmax, label)
AddBookKeepingOperators(train_model)


validation_model = model_helper.ModelHelper(name = "cifar10_validation", arg_scope = arg_scope, init_params = False)
data, label = AddInput(validation_model, batch_size = 1000, db = os.path.join(data_path, 'cifar_validation.minidb'), db_type = "minidb")
softmax = AddNetModel(validation_model, data)
AddAccuracy(validation_model, softmax, label)

test_model = model_helper.ModelHelper(name = "cifar10_test", arg_scope = arg_scope, init_params = False)
data, label = AddInput(test_model, batch_size = 1000 , db = os.path.join(data_path, 'cifar_test.minidb'), db_type = "minidb")
softmax = AddNetModel(test_model, data)
AddAccuracy(test_model, softmax, label)

deploy_model = model_helper.ModelHelper(name = "cifar10-deploy", arg_scope = arg_scope, init_params = False)
AddNetModel(deploy_model, "data")

### Dump the output files into disk.
with open(os.path.join(output_path, "train_net.pbtxt"), 'w') as fid:
	fid.write(str(train_model.net.Proto()))
with open(os.path.join(output_path, "train_init_net.pbtxt"), 'w') as fid:
	fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(output_path, "validation_net.pbtxt"), 'w') as fid:
	fid.write(str(validation_model.net.Proto()))
with open(os.path.join(output_path, "validation_init_net.pbtxt"), 'w') as fid:
	fid.write(str(validation_model.param_init_net.Proto()))
with open(os.path.join(output_path, "test_net.pbtxt"), 'w') as fid:
	fid.write(str(test_model.net.Proto()))
with open(os.path.join(output_path, "test_init_net.pbtxt"),'w') as fid:
	fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(output_path, "deploy_net.pbtxt"), 'w') as fid:
	fid.write(str(deploy_model.net.Proto()))
print("INFO: Protocol buffers have been saved to the disk.")


#### Executing the training.

print ("INFO: Executing on the training dataset")
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite = True)

total_iters = 3000
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)

for i in range(total_iters):
	workspace.RunNet(train_model.net)
	accuracy[i] = workspace.FetchBlob('accuracy')
	loss[i] = workspace.FetchBlob('loss')
	print "Iteration : {}".format(i)
	print "The accuracy of training : {}".format(accuracy[i])
	print "The loss of the training : {}".format(loss[i])


### Validation dataset accuracy.
print ("INFO: Executing on the validation dataset.")
workspace.RunNetOnce(validation_model.param_init_net)
workspace.CreateNet(validation_model.net, overwrite = True)
validation_accuracy = np.zeros(10000)

for i in range(10000):
	workspace.RunNet(validation_model.net.Proto().name)
	validation_accuracy[i] = workspace.FetchBlob('accuracy')

print ("RESULT: Overall validation accuracy is : %f" % validation_accuracy.mean())
		
