import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from time import time
from time import sleep
import os
import re
from numpy import array
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sn

from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers.merge import add

# Arrays/Lists for Train Data and Train labels
DataSet = []
X = []
Y = []


def load_DataSet():
    output_dir = "C:\\Users\\Afnan\\Desktop\\Dataset\\2nd One\\Processed_Digits\\"

    files = os.listdir(output_dir)
    print('No of files: ' + str(len(files)))

    # Reading all the image files from the directory
    for filename in os.listdir(output_dir):
        img = cv2.imread(output_dir+filename,0)
        label = re.split('_',filename)
        DataSet.append([label[0],img])       

    # Separting the labels and image data
    for label,img in DataSet:
        X.append(label)
        Y.append(img)
    
def CNN_function():
    x_train_label = []
    x_test_label = []
    x_predict_label = []

    x_train_dataset = []
    x_test_dataset = []
    x_predict_dataset = []

    X_label = pd.Categorical(X)
    categories = X_label.categories
    X_label = X_label.codes

    # Split the data into train and test dataset
    x_train_label, x_test_label, x_train_dataset, x_test_dataset = train_test_split(X_label,Y,test_size=0.19, random_state=3)

    # Split the data train data into further training labels and prediction dataset
    x_train_label, x_predict_label, x_train_dataset, x_predict_dataset = train_test_split(x_train_label, x_train_dataset,test_size=0.23, random_state=3)

    x_train_nmpy = np.array(x_train_label)
    x_test_nmpy = np.array(x_test_label)
    x_predict_nmpy = np.array(x_predict_label)

    # Normalize the pixel values of the train data and test data
    x_train_dataset = tf.keras.utils.normalize(x_train_dataset, axis = 1)
    x_test_dataset = tf.keras.utils.normalize(x_test_dataset, axis = 1)
    x_predict_dataset = tf.keras.utils.normalize(x_predict_dataset, axis = 1)

    # Reshaping the dataset for input to neural network
    x_train_dataset =  x_train_dataset.reshape((x_train_dataset.shape[0], 28, 28, 1)).astype('float32')  
    x_test_dataset =  x_test_dataset.reshape((x_test_dataset.shape[0], 28, 28, 1)).astype('float32')  
    x_predict_dataset =  x_predict_dataset.reshape((x_predict_dataset.shape[0], 28, 28, 1)).astype('float32') 

    # Creating a Sequential Model for Neural Network
    # 2 Convolution Layers with 30,15 filter and Kernel of 6x6 and 3x3 
    # Pooling Layer 2 with (2x2)
    # 2 Dense layer
    # Flatten Layer

    model = Sequential()
    model.add(Conv2D(30, (6, 6), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compilation of given model
    model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    num_epochs = 30

    # Training our model
    Model1 = model.fit(x_train_dataset, x_train_nmpy, epochs=num_epochs, validation_data =(x_test_dataset, x_test_nmpy))

    plt.plot(Model1.history['accuracy'])
    plt.plot(Model1.history['val_accuracy'])
    
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    

    # Predictions and Testing Part

    predictions = model.predict(x_predict_dataset)
    
    pred = []

    for i in range(0,len(x_predict_label)):
        pred.append(np.argmax(predictions[i]))

    total = len(pred)
    n_p = 0
    for x in range(0,len(pred)):
        if (int(pred[x]) == int(x_predict_label[x])):
            n_p = n_p + 1

    
    print("The accuracy of test set: " + str((n_p/total)*100))

    a = x_predict_label.tolist()

    cm = confusion_matrix(a, pred)
    print(cm)

    # Plotting the confusion matrix
    df_cm = pd.DataFrame(cm, range(10), range(10))
    sn.set(font_scale=1.2) #for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    plt.show()



if __name__ == "__main__":
    load_DataSet()
    CNN_function()
