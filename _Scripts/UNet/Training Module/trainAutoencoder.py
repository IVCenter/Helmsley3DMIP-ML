import time
import datetime
from autoencoderModel import *
from autoencoderData import *
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import TensorBoard
# Check the currently available GPUs
print(device_lib.list_local_devices())


'''
Get time
'''
now = time.time()
time_stamp = datetime.datetime.fromtimestamp(now).strftime('_%m_%d_%H_%M')


'''
This script train the model using the PNG images and labels, and save the model as hdf5 file
'''

image_folder = 'mri_images_16_17'
label_folder = 'mri_images_16_17'
save_folder = 'model_archive'
model_name = 'autoencoder'
log_folder = 'log'

model_name = model_name + time_stamp

'''
Set the parameters and the model type for the training
'''
data_gen_args = dict(rotation_range=0.2,
					width_shift_range=0.05,
					height_shift_range=0.05,
					shear_range=0.05,
					zoom_range=0.05,
					horizontal_flip=True,
					fill_mode='nearest')

save_path = save_folder + '/' + model_name + '.hdf5'

myGene = trainGenerator(4,'Datasets',image_folder,label_folder,data_gen_args,save_to_dir = None, target_size = (512, 512))

model = autoencoder2((512, 512, 1), 2000)

model_checkpoint = ModelCheckpoint(save_path, monitor='loss',verbose=1, save_best_only=True)
tensorboard_callback = TensorBoard(log_dir=log_folder,histogram_freq=1)

'''
The training starts here.
'''
model.fit_generator(myGene,steps_per_epoch=1024, epochs=32,callbacks=[model_checkpoint, tensorboard_callback])







