import time
import datetime
from model import *
from data import *
from tensorflow.python.client import device_lib

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

image_folder = 'mri_image_2016'
label_folder = 'mri_label_2016'
save_folder = 'model_checkpoint'
model_name = 'colon'

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

myGene = trainGenerator(2,'Datasets',image_folder,label_folder,data_gen_args,save_to_dir = None)

model = unet()
# model = unet_lrelu()

model_checkpoint = ModelCheckpoint(save_path, monitor='loss',verbose=1, save_best_only=True)

'''
The training starts here.
'''
model.fit_generator(myGene,steps_per_epoch=20000,epochs=5,callbacks=[model_checkpoint])







