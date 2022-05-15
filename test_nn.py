import os
import pandas as pd
from PIL import Image

import numpy as np

from src.activation import relu, softmax
from src.cost import softmax_cross_entropy
from src.optimizer import adam

from src.layers.conv import Conv
from src.layers.dense import Dense
from src.layers.pooling import Pool
from src.layers.dropout import Dropout
from src.layers.flatten import Flatten

from src.nn import NeuralNetwork

trainpaths = os.listdir("./data/fingers/train")
testpaths = os.listdir("./data/fingers/test")

train_str = "./data/fingers/train/"
trainpaths = ["./data/fingers/train/" + p for p in trainpaths]

test_str = "./data/fingers/test/"
testpaths = ["./data/fingers/test/" + p for p in testpaths]

df_train = pd.DataFrame(trainpaths, columns=['Filepath'])
df_train['set'] = 'train'
df_test = pd.DataFrame(testpaths, columns=['Filepath'])
df_test['set'] = 'test'

image_df = pd.concat([df_train,df_test])

image_df['Label'] = image_df['Filepath'].apply(lambda x: x[-6])
image_df['Label_LR'] = image_df['Filepath'].apply(lambda x: x[-5])

train_df = image_df[image_df['set'] == 'train']
test_df = image_df[image_df['set'] == 'test']

Y_train = np.array(train_df['Label'])
Y_test = np.array(test_df['Label'])

x_train = np.asarray([])
y_train = Y_train.astype(np.int32)
x_test  = np.asarray([])
y_test  = Y_test.astype(np.int32)

with np.load("./data/fingers/xtrain.npz") as data:
    x_train = data['arr_0']

with np.load("./data/fingers/xtest.npz") as data:
    x_test = data['arr_0']

def one_hot(y, num_classes=6):
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot


def preprocess(x_train, y_train, x_test, y_test):
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_train = one_hot(y_train)
    x_train /= 255.
    x_test /= 255.
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test)

cnn = NeuralNetwork(
        input_dim=(128,128,1),
        layers=[
            Conv(3, 1, 32, activation=relu),
            Pool(2, 2),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation=relu),
            Dropout(0.1),
            Dense(6, activation=softmax)
            ],
        cost_function=softmax_cross_entropy,
        optimizer=adam
)

cnn.train(x_train=x_train, y_train=y_train,
          batch_size=256,
          num_epochs=10,
          learning_rate=0.001,
          validation_data=(x_test, y_test))