import time
import datetime
from model import *
from model_blocks import uNetModel
from data import *
from test_model import *
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

image_folder = 'mri_image_16_17_66_autoencoder'
label_folder = 'mri_label_16_17_66'
save_folder = 'model_archive'
model_name = 'colon'
log_folder = "log"

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

myGene = trainGenerator(8,'Datasets',image_folder,label_folder,data_gen_args,save_to_dir = None)


model, cpuModel = unet_batch_norm()

#model_checkpoint = ModelCheckpoint(save_path, monitor='loss',verbose=1, save_best_only=True)
tensorboard_callback = TensorBoard(log_dir=log_folder,histogram_freq=2, write_grads=True, write_images=True)

'''
The training starts here.
'''
model.fit_generator(myGene,steps_per_epoch=8000,epochs=5,callbacks=[tensorboard_callback])

if(cpuModel):
    cpuModel.save(save_path)
else:
    model.save(save_path)








