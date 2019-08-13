import os
import sys

if len(sys.argv) != 3:
	print ("ERROR: \n **You have to give two command line arguments** \n" + 
		" First arg: input folder name that contains your PNG images \n"  +
		" Second arg: the path to the hdf5 file \n");
	exit()


import time
import datetime
#from model import *
from data import *
from test_model import *
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

'''
Get time
'''
now = time.time()
time_stamp = datetime.datetime.fromtimestamp(now).strftime('_%m_%d_%H')



input_path = str(sys.argv[1])
input_folder = os.path.basename(os.path.normpath(input_path))

output_folder = input_folder + time_stamp
output_path = './newTests/bladder/' + output_folder

num_images = len([name for name in os.listdir(input_path) if os.path.isfile(os.path.join(input_path,name))])

# print ("num_images = " + str(num_images))

path_to_the_hdf5_model = str(sys.argv[2])

'''
Run the Test scripts
'''
testGene = testGenerator(input_path,num_images)
model, cpuModel = unet_batch_norm()
model.load_weights(path_to_the_hdf5_model)

'''
Get the single gpu weight if the model is trained with multi-gpu model
This testing only uses one gpu 
'''

#single_gpu_model = model.layers[-2]
#single_gpu_model.save("./temp.hdf5")

#results = single_gpu_model.predict_generator(testGene,num_images,verbose=1)

results = model.predict_generator(testGene,num_images,verbose=1)
'''
Save the results
'''
os.system("mkdir " + output_path)
saveResult(output_path,results)
#os.system("rm ./temp.hdf5")


