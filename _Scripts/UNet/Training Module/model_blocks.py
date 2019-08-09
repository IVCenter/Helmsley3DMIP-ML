import numpy as np 
import os
import tensorflow as tf   
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import * 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Input, Conv2D, MaxPooling2D,\
Dropout, UpSampling2D, concatenate
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.initializers import glorot_uniform 

def twoConvBlockLeaky(inputs, nChannel:int, alpha:float=0.05, kernel_initializer:str='glorot_normal'):
    conv1 = Conv2D(nChannel, 3, activation = 'linear', padding = 'same', kernel_initializer = kernel_initializer)(inputs)
    conv1 = LeakyReLU(alpha)(conv1)
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv2D(nChannel, 3, activation = 'linear', padding = 'same', kernel_initializer = kernel_initializer)(conv1)
    conv2 = LeakyReLU(alpha)(conv2)
    conv2 = BatchNormalization()(conv2)
    return conv2

def threeConvBlockLeaky(inputs, nChannel:int, alpha:float=0.05, kernel_initializer:str='glorot_normal'):
    conv1 = Conv2D(nChannel, 3, activation = 'linear', padding = 'same', kernel_initializer = kernel_initializer)(inputs)
    conv1 = LeakyReLU(alpha)(conv1)
    conv1 = BatchNormalization()(conv1)
    
    conv2 = Conv2D(nChannel, 3, activation = 'linear', padding = 'same', kernel_initializer = kernel_initializer)(conv1)
    conv2 = LeakyReLU(alpha)(conv2)
    conv2 = BatchNormalization()(conv2)
    
    conv3 = Conv2D(nChannel, 3, activation = 'linear', padding = 'same', kernel_initializer = kernel_initializer)(conv2)
    conv3 = LeakyReLU(alpha)(conv3)
    conv3 = BatchNormalization()(conv3)
    return conv3

def concatThreeConvBlock(inputs, toContac, nChannel:int, kernel_initializer:str='glorot_normal'):
    conv1 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_initializer)(inputs)
    conv1 = BatchNormalization()(conv1)
    merge1 = concatenate([toContac,conv1], axis = 3)
    
    conv2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_initializer)(merge1)
    conv2 = BatchNormalization()(conv2)
    
    conv3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_initializer)(conv2)
    conv3 = BatchNormalization()(conv3)
    
    return conv3

def compileModel(inputs, lastLayer, G: int=1):
    
    if(G == 1):
        model = Model(inputs = inputs, outputs = lastLayer)
        print(model.summary()) 
    else:
        with tf.device("/cpu:0"):
            model = Model(inputs = inputs, outputs = lastLayer)
            print(model.summary()) 
        
        model = multi_gpu_model(model, gpus=G)
        
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

def uNetModel(input_size = (256,256,1)):
    inputs = Input(input_size)
    
    conv1 = twoConvBlockLeaky(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = twoConvBlockLeaky(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = twoConvBlockLeaky(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = twoConvBlockLeaky(pool3, 512)
    drop4 = Dropout(0.5)(conv4) 
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = twoConvBlockLeaky(pool4, 1024)
    drop5 = Dropout(0.5)(conv5) 
    up5 = UpSampling2D(size = (2,2))(drop5)
    
    conv6 = concatThreeConvBlock(up5, drop4, 512)
    up6 = UpSampling2D(size = (2,2))(conv6)
    
    conv7 = concatThreeConvBlock(up6, conv3, 256)
    up7 = UpSampling2D(size = (2,2))(conv7)
    
    conv8 = concatThreeConvBlock(up7, conv2, 128)
    up8 = UpSampling2D(size = (2,2))(conv8 )
    
    conv9 = concatThreeConvBlock(up8, conv1, 64)
    conv9_2 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv9)
    conv9_2 = BatchNormalization()(conv9_2)
    
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9_2)
    
    return compileModel(inputs, conv10, G=1)