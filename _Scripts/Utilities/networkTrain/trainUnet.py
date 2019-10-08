import time
import datetime
from data import *
from test_model import *
from tensorflow.python.client import device_lib

def trainUnet(colorDict, input_size):

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

	image_folder = 'trainingImage'
	label_folder = 'trainingMask'
	save_folder = 'model_archive'
	model_name = 'segmentation'

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
	'''
	data_gen_args = dict(rotation_range=0,
						width_shift_range=0,
						height_shift_range=0,
						shear_range=0,
						zoom_range=0,
						horizontal_flip=False,
						fill_mode='nearest')
	'''
	save_path = save_folder + '/' + model_name + '.hdf5'

	myGene = trainGenerator(8,'tmp',image_folder,label_folder,data_gen_args, colorDict, save_to_dir = None, image_color_mode = 'rgb', mask_color_mode = 'rgb', target_size=(input_size[:2]))


	model, cpuModel = unet(numLabels=len(colorDict), input_size=input_size)

	'''
	The training starts here.
	'''
	model.fit_generator(myGene,steps_per_epoch=4096,epochs=5)

	if(cpuModel):
		cpuModel.save(save_path)
	else:
		model.save(save_path)








