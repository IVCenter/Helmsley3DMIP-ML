import sys
import os

verbose = 1

def DebugLog(output_string):
	if verbose == 1:
		print ("(DebugLog): " + output_string)

if len(sys.argv) != 3 and len(sys.argv) != 4:
	print ("ERROR: \n **You have to give two or three command line arguments** \n" + 
		" First arg: input folder that contains your dicom images \n" + 
		" Second arg: output folder for the predicted PNG masks \n" + 
		" Third arg (optional): the path to the other U-Net Model you" + 
		" want to use for prediction.\n (default is unet_colon.hdf5, make sure you have it in this directory)\n")
	exit()

filename = str(sys.argv[0])
path_to_the_dicom_images = str(sys.argv[1])
path_to_the_final_output_folder = str(sys.argv[2])

if len(sys.argv) == 4:
	path_to_the_hdf5_model = str(sys.argv[3])
else:
	path_to_the_hdf5_model = "unet_colon.hdf5"



# read over the directory, count the number of files.  
# parse two strings for input out put  
# dicom to mask directly.  
# save as numpy array makes better  
# run python on the same heap-space.  


################################ Conversion ########################
DebugLog ("Importing resources for converting images...")
import numpy as np
import png
import pydicom
from os import listdir
from os.path import isfile, join

os.system("mkdir ./Converted")
input_folder_path = path_to_the_dicom_images
output_folder_path = "./Converted"

# all_dcms = [f for f in listdir(input_folder_path) if isfile(join(input_folder_path,f)) if f.endswith(".dcm")]
all_dcms = [pydicom.read_file(input_folder_path + '/' + f) for f in listdir(input_folder_path) if isfile(join(input_folder_path,f)) if f.endswith(".dcm")]

all_dcms.sort(key = lambda x: int(x[0x20, 0x32][1]))

dcm_count = 0

if len(all_dcms) == 0:
	DebugLog ("Error: No dcm files found")
	exit()

# print(" [ Found " + str(len(all_dcms)) + " images... ]")
# print(" [ Started converting... ]")
# print(' ', end='')
# print('[', end='')

curr_progress = 0
prev_progress = 0

DebugLog ("Started converting dicom to PNGs ...")
sys.stdout.write("Progress: [")
sys.stdout.flush()

for dcm in all_dcms:
	
	curr_progress = dcm_count * 1.0 / len(all_dcms)

	# dcm_path = input_folder_path + "/" + dcm
	ds = dcm

	shape = ds.pixel_array.shape

	# Convert to float to avoid overflow or underflow losses.
	image_2d = ds.pixel_array.astype(float)

	# Rescaling grey scale between 0-255
	image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

	# Convert to uint
	image_2d_scaled = np.uint8(image_2d_scaled)

	output_name = output_folder_path + "/" + str(dcm_count) + ".png"

	# Write the PNG file
	with open(output_name, 'wb') as png_file:
		w = png.Writer(shape[1], shape[0], greyscale=True)
		w.write(png_file, image_2d_scaled)

	dcm_count += 1

	if curr_progress - prev_progress > 0.02:
		sys.stdout.write("=")
		sys.stdout.flush()
		prev_progress = curr_progress

sys.stdout.write("]\n")
sys.stdout.flush()
DebugLog (" [ Done! Converted "+ str(dcm_count) + " images ]")
DebugLog ("converted PNGs saved to " + output_folder_path)

#******************************************************************#

# retrieve the number of images
number_of_dicoms = len(all_dcms)

################################ Prediction ########################

DebugLog ("Importing resources for prediction")
from model import *
from data import *
DebugLog ("predicting masks....")

path_to_the_mri_images = output_folder_path
path_to_the_model = path_to_the_hdf5_model
path_to_the_output_folder = path_to_the_final_output_folder
number_of_images = number_of_dicoms

testGene = testGenerator(path_to_the_mri_images, num_image=number_of_images)
model = unet()
model.load_weights(path_to_the_model)
results = model.predict_generator(testGene,number_of_images,verbose=1)
saveResult(path_to_the_output_folder,results) 

DebugLog ("Cleaning temp folders")
os.system("rm -rf ./Converted")

DebugLog ("\n\tCompleted!")





