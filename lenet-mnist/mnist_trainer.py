import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe


from caffe2.python import core, model_helper, workspace, brew

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
print("Necessities imported!")

# This section preps your image and test set in a lmdb database
def DownloadResource(url, path):
    '''Downloads resources from s3 by url and unzips them to the provided path'''
    import requests, zipfile, StringIO
    print("Downloading... {} to {}".format(url, path))
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)
    print("Completed download and extraction.")


current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks')
data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
db_missing = False

if not os.path.exists(data_folder):
    os.makedirs(data_folder)   
    print("Your data folder was not found!! This was generated: {}".format(data_folder))

# Look for existing database: lmdb
if os.path.exists(os.path.join(data_folder,"mnist-train-nchw-lmdb")):
    print("lmdb train db found!")
else:
    db_missing = True

if os.path.exists(os.path.join(data_folder,"mnist-test-nchw-lmdb")):
    print("lmdb test db found!")
else:
    db_missing = True

# attempt the download of the db if either was missing
if db_missing:
    print("one or both of the MNIST lmbd dbs not found!!")
    db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
    try:
        DownloadResource(db_url, data_folder)
    except Exception as ex:
        print("Failed to download dataset. Please download it manually from {}".format(db_url))
        print("Unzip it and place the two database folders here: {}".format(data_folder))
        raise ex

if os.path.exists(root_folder):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree(root_folder)

os.makedirs(root_folder)
workspace.ResetWorkspace(root_folder)

print("training data folder:" + data_folder)
print("workspace root folder:" + root_folder)


def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label

def AddLeNetModel(model, data):
	conv1 = brew.conv(model, data, 'conv1', dim_in = 1, dim_out = 20, kernel = 5)
	pool1 = brew.max_pool(model, conv1, 'pool1', kernel = 2, stride = 2)
	conv2 = brew.conv(model, pool1, 'conv2', dim_in = 20, dim_out = 50, kernel = 5)
	pool2 = brew.max_pool(model, conv2, 'pool2', kernel = 2, stride = 2)
	fc3 = brew.fc(model, pool2, 'fc3', dim_in = 50 * 4 * 4, dim_out = 500)
	fc3 = brew.relu(model, fc3, fc3)
	pred = brew.fc(model, fc3, 'pred', 500, 10)
	softmax = brew.softmax(model, pred, 'softmax')
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
	LR = model.LearningRate(ITER, "LR", base_lr = -0.1, policy = "step", stepsize=1, gamma = 0.999)
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

#### Actual datainput, computation, training and bookkeeping.
arg_scope = {"order" : "NCHW"}
train_model = model_helper.ModelHelper(name = "mnist_train", arg_scope = arg_scope)
data, label = AddInput(train_model, batch_size = 64, db = os.path.join(data_folder, 'mnist-train-nchw-lmdb'), db_type = 'lmdb')
softmax = AddLeNetModel(train_model, data)
AddTrainingParameters(train_model, softmax, label)
AddBookKeepingOperators(train_model)

test_model = model_helper.ModelHelper(name = "mnist-test", arg_scope = arg_scope, init_params = False)
data, label = AddInput(test_model, batch_size = 100, db = os.path.join(data_folder, 'mnist-train-nchw-lmdb'), db_type = 'lmdb')
softmax = AddLeNetModel(test_model, data)
AddAccuracy(test_model, softmax, label)

deploy_model = model_helper.ModelHelper(name = "mnist-deploy", arg_scope = arg_scope, init_params = False)
AddLeNetModel(deploy_model, "data")


#### Dump the protocol buffers to the disk.
with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
    fid.write(str(deploy_model.net.Proto()))
print("Protocol buffers files have been created in your root folder: " + root_folder)

#### Actual training procedure.
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite = True)

total_iters = 400
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)

for i in range(total_iters):
	workspace.RunNet(train_model.net)
	accuracy[i] = workspace.FetchBlob('accuracy')
	loss[i] = workspace.FetchBlob('loss')
	print "Iteration : {}".format(i)
	print "The accuracy for the model is : {} ".format( accuracy[i])
	print "The loss for thee model is : {}".format(loss[i])



##### Test dataset.
print ("INFO: Starting the testing procedure")

workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite = True)
test_accuracy = np.zeros(100)

for i in range(100):
	workspace.RunNet(test_model.net.Proto().name)
	test_accuracy[i] = workspace.FetchBlob('accuracy')
	
print ("RESULT: Overall Test accuracy is : %f" % test_accuracy.mean())

'''
#### Save the deploy model into the disk.
pe_meta = pe.PredictorExportMeta(predict_net = deploy_model.net.Proto(), parameters = [str(b) for b in deploy_model.params], inputs = ["data"],
					outputs = ["softmax"] )
pe.save_to_db("minidb", os.path.join(root_folder, "mnist_model.minidb", pe_meta)

'''
