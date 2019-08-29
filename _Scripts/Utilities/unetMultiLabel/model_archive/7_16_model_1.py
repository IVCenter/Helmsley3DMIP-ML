def autoencoder1_2(input_size=(512, 512, 1)):

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


