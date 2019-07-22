import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
G = 2
def autoencoder(input_size=(512, 512, 1)):

    input_img = Input(shape=input_size)  # adapt this if using `channels_first` image data format

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    
    if(G == 1):
        model = Model(input_img, decoded)
    else:
        with tf.device("/cpu:0"):
            model = Model(input_img, decoded)
    
        model = multi_gpu_model(model, gpus=G)
        
    print(model.summary()) 
    model.compile(optimizer='RMSprop', loss='mean_squared_error')
        
    return model

def autoencoder1_2(input_size=(512, 512, 1)):

    input_img = Input(shape=input_size)  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (3, 3), activation='linear', padding='same')(input_img)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='linear', padding='same')(x)
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
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(0.4)(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    
    if(G == 1):
        model = Model(input_img, decoded)
    else:
        with tf.device("/cpu:0"):
            model = Model(input_img, decoded)
    
        model = multi_gpu_model(model, gpus=G)
        
    print(model.summary()) 
    model.compile(optimizer='RMSprop', loss='mean_squared_error')
        
    return model

def autoencoder2(input_size, nz):

    input_img = Input(shape=input_size)  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu',  padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3),  activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (3, 3),  activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1024, (3, 3),  activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(2048, (3, 3), activation='relu',  padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(2048, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    '''
    # at this point the representation is (4, 4, 1024) i.e. 16384-dimensional
    x = Reshape((-1, 2048*4*4))(x)
    encoded = Dense(nz, activation='sigmoid')(x)

    # Begin decoding
    
    x = Dense(4*4*2048)(encoded)
    x = Reshape((4, 4, 2048))(x)
    '''
    x = Conv2D(2048, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(2048, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, 1, activation='sigmoid', padding='same')(x)
    
    if(G == 1):
        model = Model(input_img, decoded)
    
    else:
        with tf.device("/cpu:0"):
            model = Model(input_img, decoded)
    
        model = multi_gpu_model(model, gpus=G)
        
    #model.compile(optimizer='adam', loss='binary_crossentropy')
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(model.summary()) 
     
    return model
