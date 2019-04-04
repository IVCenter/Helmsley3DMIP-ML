import os

if len(sys.argv) != 3:
	print ("ERROR: \n **You have to give two command line arguments** \n" + 
		" First arg: input folder name that contains your PNG images \n"  +
		" Second arg: the path to the hdf5 file \n");
	exit()


import time
import datetime
from model import *
from data import *
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

'''
Get time
'''
now = time.time()
time_stamp = datetime.datetime.fromtimestamp(now).strftime('_%m_%d_%H_%M')



input_path = str(sys.argv[1])
input_folder = os.path.basename(os.path.normpath(input_path))

output_folder = input_folder + time_stamp
output_path = './Tests/' + output_folder

path_to_the_hdf5_model = str(sys.argv[2])

'''
Run the Test scripts
'''
testGene = testGenerator(input_folder)
model = unet()
model.load_weights(path_to_the_hdf5_model)
results = model.predict_generator(testGene,5,verbose=1)

'''
Save the results
'''
os.system("mkdir " + output_path)
saveResult(output_path,results)


