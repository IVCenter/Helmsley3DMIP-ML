# ***************************************************************
#
# Convert all dcm files in ToConvet folder to png and store or 
# overwrite in the Converted folder with name 0.png, 1.png, etc.
#
# ***************************************************************

import numpy as np
import png
import pydicom
from os import listdir
from os.path import isfile, join

input_folder_path = "./ToConvert"
output_folder_path = "./Converted"

all_dcms = [f for f in listdir(input_folder_path) if isfile(join(input_folder_path,f)) if f.endswith(".dcm")]

dcm_count = 0

if len(all_dcms) == 0:
	print ("No dcm files found")
	exit()

print(" [ Found " + str(len(all_dcms)) + " images... ]")
print(" [ Started converting... ]")
print(' ', end='')
print('[', end='')

curr_progress = 0
prev_progress = 0

for dcm in all_dcms:
	
	curr_progress = dcm_count * 1.0 / len(all_dcms)

	dcm_path = input_folder_path + "/" + dcm

	ds = pydicom.dcmread(dcm_path)

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
		print('=', end='')
		prev_progress = curr_progress

print ("]\n [ Done! Converted "+ str(dcm_count) + " images ]")
