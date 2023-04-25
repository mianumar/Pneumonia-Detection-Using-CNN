# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:30:38 2023

@author: umerr
"""

import cv2
import numpy as np
import imageio as IO
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import random
# import torch

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf


import tkinter
from tkinter import filedialog
from tkinter import simpledialog

    
    
seed =232
img_dims = 200
epochs = 15   #Number of times to itrate Neural Network Fully Connected Layers
batch_size = 64  #Bunch of images to send them in group to Models and NN FCLayers

np.random.seed(seed)
input_path = 'D:\\PUCIT MPhill CS\\Semester 1\\Medical Image Analysis\\Pneumonia Detection\\chest_xray\\chest_xray\\'
fig, ax = plt.subplots(2,3, figsize=(15,15))
ax= ax.ravel()
plt.tight_layout()



#Displaying images
for i, set in enumerate(['train', 'val' , 'test']):
    set_path = input_path + set
    
    normal_path = set_path + '\\NORMAL\\'
    pneumonia_path = set_path + '\\PNEUMONIA\\'
    
    
    ax[i].imshow(plt.imread(normal_path + os.listdir(set_path+'\\NORMAL')[0]),cmap='gray')
    ax[i].set_title('Set: {}, Condition: Normal'.format(set))
    ax[i+3].imshow(plt.imread(pneumonia_path + os.listdir(set_path+'\\PNEUMONIA')[0]),cmap='gray')
    ax[i+3].set_title('Set: {}, Condition: PNEUMONIA'.format(set))
    

#Displaying total dataset images in each folder
for set in ['test', 'train', 'val']:
    normal_path = input_path + set + '\\NORMAL\\'
    pneumonia_path = input_path + set + '\\PNEUMONIA\\'
    print('Set: {}, Normal Images: {}, Pneumonia Images: {}'.format(
        set, len(os.listdir(normal_path)), len(os.listdir(pneumonia_path))))



"""
Image Augmentation
1. Rescaling the Images
2. Zoom the Images
3. Flipping images vertically
We apply Augmentation on both "Train" and "Test" Dataset
""" 
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)
test_val_datagen = ImageDataGenerator(rescale=1./255)
print("Train Data Gen" , train_datagen)
print("Test Data Gen" , test_val_datagen)

"""
Generating Training Data Set
""" 
train_gen = train_datagen.flow_from_directory(directory=input_path+'train', target_size=(img_dims, img_dims), batch_size=batch_size, class_mode='binary', shuffle=True)
# print(train_datagen)

"""
Generating Testing Data Set
""" 
test_gen = test_val_datagen.flow_from_directory(directory=input_path+'test', target_size=(img_dims, img_dims), batch_size=batch_size, class_mode='binary', shuffle=True)
# print(test_gen)

"""
Generating Test Data as Array
that will contain test_label array as
0 for Normal Image
1 for Pneumonia
""" 
def process_data(img_dims, batch_size):
    test_data = []
    test_labels = []
    
    for cond in ['//NORMAL//' , '//PNEUMONIA//']:
        for img in (os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path+'test'+cond+img)
            img = cv2.resize(img, (img_dims,img_dims))
            img = np.dstack([img,img,img])
            img = img.astype('float32') / 255
            if cond == '//NORMAL//':
                label = 0
            elif cond == '//PNEUMONIA//':
                label = 1
            test_data.append(img)
            test_labels.append(label)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    # print("Test Data: " , test_data , "Test Labels: " , test_labels)
    return test_data, test_labels

# Data Process Function call
test_data, test_labels = process_data(img_dims, batch_size)
# print("Test Data O: " , test_data , "Test Labels O: " , test_labels)

"""
Building the Convolution Model
Step 1: 2 Convolotional Layer
Step 2: Max Pooling and Batch Normalization
Step 3: Flatten Layer
Step 4: 2 Fully Connected Layers

Activation Function Used
For Hidden Layer: ReLU Function
For Output Layer: Sigmoid Function (For Binary Classification)

Optimizer: Adam
Loss Function: Cross-entropy
""" 


"""
Step 1
First Convolutional Layer
"""
inputs = Input(shape=(img_dims, img_dims, 3))

# First Convolution Block
x = Conv2D(filters=16, kernel_size=(3, 3) , activation='relu', padding='same')(inputs)
x = Conv2D(filters=16, kernel_size=(3, 3) , activation='relu', padding='same')(x)

# Step 1 Pooling Layer
x = MaxPool2D(pool_size=(2,2))(x)


"""
Step 2
Second Convolutional Layer
"""
x = SeparableConv2D(filters=32, kernel_size=(3, 3) , activation='relu', padding='same')(x)
x = SeparableConv2D(filters=32, kernel_size=(3, 3) , activation='relu', padding='same')(x)
x = BatchNormalization()(x)

# Step 2 Pooling Layer
x = MaxPool2D(pool_size=(2,2))(x)

# Dropputs to reduce Overfitting
x = Dropout(rate = 0.2)(x)


"""
Step 3
Fully Connected Layer
"""
x = Flatten()(x)

"""
Step 4
Hidden Layers
"""
x = Dense(units = 128, activation='relu')(x)        #Units is num of nodes
x = Dropout(rate=0.5)(x)
x = Dense(units = 64, activation='relu')(x)
x = Dropout(rate=0.25)(x)           #Dropout to control overfitting

"""
Step 5
Output Layer
"""
output = Dense(units = 1, activation='sigmoid')(x)



# # Input Layer
# input_shape = (img_dims, img_dims, 3)
# inputs = Input(input_shape)
# Output Layer
# output = Dense(units = 1, activation='sigmoid')(x)


# Creating Model and Cimpling
model = Model(inputs = inputs, outputs = output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#adam stands for Adaptive momentum Estimation replacement 
model.summary()



"""
Callbacks
"""
checkpoint = ModelCheckpoint(filepath = 'best_weights.hdf11', save_best_only=True, save_weights_only=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss' , min_delta=0.1, patience=1, mode='min')


"""
Fitting Model
"""
hist = model.fit(train_gen, steps_per_epoch=train_gen.samples // batch_size, epochs=epochs, validation_data=test_gen, validation_steps=test_gen.samples // batch_size, callbacks=[checkpoint, lr_reduce])



"""
Visualizing Loss and Accuracy Plots
"""
fig, ax = plt.subplots(1, 2, figsize=(10,3))
# print("ERROR : " , plt.subplots(1, 2, figsize=(10,3)))
ax = ax.ravel()
for i, met in enumerate(['accuracy' , 'loss']) :
    ax[i].plot(hist.history[met])
    ax[i].plot(hist.history['val_' + met])
    ax[i].set_title('Model{}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train' , 'val'])



"""
Accuracy and Confussion Matrix
"""
preds = model.predict(test_data)

acc = accuracy_score(test_labels, np.round(preds))*100
cm = confusion_matrix(test_labels, np.round(preds))
tn, fp, fn, tp = cm.ravel()

print("Confussion Matrix -------------------")
print(cm)



"""
Performance Analysic
"""
print('\nTest Metrics ---------------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))


# Predict labels for test data
pred_labels = model.predict(test_data)

# Map the predicted labels to actual labels
pred_labels = np.round(pred_labels).flatten()

# Print the number of positive and negative cases
print("Number of Positive Cases (Pneumonia): ", np.sum(pred_labels==1))
print("Number of Negative Cases (Normal): ", np.sum(pred_labels==0))



print("===========================???????=================")
