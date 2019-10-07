import tensorflow as tf
import tensorflow.keras as keras    
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import * 
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.initializers import glorot_uniform 
from tensorflow.keras import regularizers

# This is the number of GPU you want to use
G = 1  
alpha=0.05
beta = 0.01
def unet(numLabels:int, pretrained_weights = False,input_size = (256,256,1)):

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(inputs)
    conv1 = LeakyReLU(alpha)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv1)
    conv1 = LeakyReLU(alpha)(conv1)
    conv1 = BatchNormalization()(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(pool1)
    conv2 = LeakyReLU(alpha)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv2)
    conv2 = LeakyReLU(alpha)(conv2)
    conv2 = BatchNormalization()(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(pool2)
    conv3 = LeakyReLU(alpha)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv3)
    conv3 = LeakyReLU(alpha)(conv3)
    conv3 = BatchNormalization()(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(pool3)
    conv4 = LeakyReLU(alpha)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv4)
    conv4 = LeakyReLU(alpha)(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(pool4)
    conv5 = LeakyReLU(alpha)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'linear', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv5)
    conv5 = LeakyReLU(alpha)(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(UpSampling2D(size = (2,2))(drop5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(UpSampling2D(size = (2,2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(numLabels, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(beta))(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(numLabels, 1, activation = 'sigmoid')(conv9)
    
    if(G == 1):
        cpuModel = None
        model = Model(inputs = inputs, outputs = conv10)
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        print(model.summary()) 
    else:
        with tf.device("/cpu:0"):
            cpuModel = Model(inputs = inputs, outputs = conv10)
            print(cpuModel.summary())
        
        model = multi_gpu_model(cpuModel, gpus=G)
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    return model, cpuModel

    #if(pretrained_weights):
    #	model.load_weights(pretrained_weights)

def autoencoder1_2(input_size=(512, 512, 1)):

    input_img = Input(shape=input_size)  # adapt this if using `channels_first` image data format

    x = Conv2D(128, (3, 3), activation='linear', padding='same')(input_img)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='linear', padding='same')(input_img)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(512, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(1024, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = SpatialDropout2D(0.5)(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    
    if(G == 1):
        cpuModel = None
        model = Model(inputs = input_img, outputs = decoded)
        model.compile(optimizer='RMSprop', loss='mean_squared_error')
        print(model.summary()) 
    else:
        with tf.device("/cpu:0"):
            cpuModel = Model(inputs = inputs, outputs = conv10)
            print(cpuModel.summary())
        
        model = multi_gpu_model(cpuModel, gpus=G)
        model.compile(optimizer='RMSprop', loss='mean_squared_error')
    return model, cpuModel
