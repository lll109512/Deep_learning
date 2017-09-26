from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Formaliziton
def Formaliziton(sample,lables):
    sample = np.transpose(sample,(3,0,1,2))
    pdseries = pd.Series(lables[:,0])
    pdseries.replace({10:0},inplace=True)
    lables = pd.get_dummies(pdseries).values
    return sample,lables

#Normalization
def Normalization(Data):
    return np.add.reduce(Data,keepdims = True,axis = 3) / (3.0 * 128) - 1.0


Train_data=loadmat('../Datas/SVHN_Dataset/train_32x32.mat')
Test_data=loadmat('../Datas/SVHN_Dataset/test_32x32.mat')

#print('Train_data sample',Train_data['X'].shape)
#print('Train_data lables',Train_data['y'].shape)
#print('Test_data sample',Test_data['X'].shape)
#print('Test_data lables',Test_data['y'].shape)

Train_data_X,Train_data_y = Formaliziton(Train_data['X'],Train_data['y'])
Test_data_X,Test_data_y = Formaliziton(Test_data['X'],Test_data['y'])

Train_data_X = Normalization(Train_data_X)
Test_data_X = Normalization(Test_data_X)

num_lables = 10
num_channels = 1
image_size = 32

if __name__ == '__main__':
    pass
