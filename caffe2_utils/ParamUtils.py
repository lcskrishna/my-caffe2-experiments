## This file contains utility funtions necessary such as adding input, collect statistics and add training parameters.
## @author: Chaitanya Sri Krishna Lolla

from caffe2.python import core, model_helper, workspace, brew
import numpy as np
import os

'''
This function adds an input layer to the model based on the data type and the db type.
@arg: model - train, validation or test model.
@arg: batch_size - batch size of the data.
@arg: db_type - the database type used in caffe2.
@arg: db - the name of the database used.
'''
def AddInputLayer(model, batch_size, db, db_type):
	data_uint8, label = model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size = batch_size, db = db, db_type = db_type)
	data = model.Cast(data_uint8, "data", to = core.DataType.FLOAT)
	data = model.StopGradient(data, data)
	return data

def AddAccuracy(model, softmax, label):
	accuracy = brew.accuracy(model, [softmax, label], "accuracy")
	return accuracy

'''
This function is used for adding the training parameters for a tester code.
@arg: model - train, validation or test model.
@arg: softmax - softmax object that is returned as the final layer in a network.
@arg: label - the output labels.
'''
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

'''
This function is used for logging purposes which exports the accuracy and loss to a file.
@arg: model - train, validation or test model.
'''
def AddBookKeepingOperators(model):
	model.Print('accuracy', [], to_file = 1)
	model.Print('loss', [], to_file = 1)
	
	for param in model.params:
		model.Summarize(param, [], to_file = 1)
		model.Summarize(model.param_to_grad[param], [], to_file = 1)
'''
This function is used to save the model into init_net.pb and predict_net.pb
@arg: model - test model
@arg : prefix - Prefix for the file names.
@arg : shape - of tensor (C, H, W) tuple is requiredtu.
'''
def SaveNet(model, prefix, tensor_shape):
	with open(prefix + '_predict_net.pb', 'wb') as f:
		f.write(model.net._net.SerializeToString())
	init_net = caffe2_pb2.NetDef()
	for param in model.params:
		blob = workspace.FetchBlob(param)
		shape = blob.shape
		op = core.CreateOperator("GivenTensorFill", [], [param], arg = [utils.MakeArgument("shape", shape), utils.MakeArgument("values", blob)])
		init_net.op.extend([op])
	init_net.op.extend([core.CreateOperator("ConstantFill", [], ["data"], shape = tensor_shape)])
	with open(prefix + '_init_net.pb', 'wb') as f:
		f.write(init_net.SerializeToString())

