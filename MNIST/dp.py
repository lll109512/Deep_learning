import numpy as np
import tensorflow as tf
import load
from sklearn.metrics import confusion_matrix


Train_data = load.Train_data
Train_labels = load.Train_labels
Test_data = load.Test_data
Test_labels = load.Test_labels


print("Train:",Train_data.shape,Train_labels.shape)
print("Test:",Test_data.shape,Test_labels.shape)
