## CIFAR 10 Classification - Custom CNN
Object classification using CIFAR 10 [dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Requirements
1. Pre-installed caffe2
2. python 2.7

## Creation of CIFAR 10 dataset for caffe2
Make sure that the dataset is present in the same folder as the file : create_cifar_db.py
Then execute the following command:

```
python create_cifar_db.py
```
This creates the training, validation and testing datasets in the format of minidb that will be saved in the current folder.

## How to run the program:
To run the training and testing of CIFAR 10 classification, execute the following command:

```
python cifar10_trainer.py > statistics.log
```

This automatically trains and tests against the dataset. The accuracy of each iteration in the training dataset is saved 
into the statistics.log.
This model is created for 3000 iterations and tested against the validation dataset.
