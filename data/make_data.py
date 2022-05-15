import os
import numpy as np
import pandas as pd
from PIL import Image

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
image_df.reset_index(drop = True, inplace = True)

image_df['Label'] = image_df['Filepath'].apply(lambda x: x[-6])
image_df['Label_LR'] = image_df['Filepath'].apply(lambda x: x[-5])

image_df = image_df.sample(frac=1,random_state=0)

train_df = image_df[image_df['set'] == 'train']
test_df = image_df[image_df['set'] == 'test']

Y_train = np.array(train_df['Label'])
Y_test = np.array(test_df['Label'])

input_array = np.zeros(shape=(len(train_df),128,128,1))
for img in range(len(train_df)):
    img_pil = Image.open(train_df['Filepath'][img])
    img_arr = np.asarray(img_pil)
    input_array[img,:,:,0] = img_arr

np.savez_compressed("./data/fingers/xtrain.npy", input_array, allow_pickle=True)
np.savez_compressed("./data/fingers/ytrain.npy", Y_train, allow_pickle=True)

input_array_test = np.zeros(shape=(len(test_df),128,128,1))
test_paths = test_df['Filepath'].reset_index(drop=True)
for img_test in range(len(test_df)):
    img_pil_test = Image.open(test_paths[img_test])
    img_arr_test = np.asarray(img_pil_test)
    input_array_test[img_test,:,:,0] = img_arr_test

np.savez_compressed("./data/fingers/xtest.npy", input_array_test, allow_pickle=True)
np.savez_compressed("./data/fingers/ytest.npy", Y_test, allow_pickle=True)