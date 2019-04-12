# ***************************************************************
#
# Convert all dcm files in ToConvet folder to png and store or 
# overwrite in the Converted folder with name 0.png, 1.png, etc.
#
# ***************************************************************

import numpy as np
from numpy import newaxis, zeros
import png
import pydicom
from os import listdir
from os.path import isfile, join

input_folder_path = "./ToConvert"
output_folder_path = "./Converted"

# all_dcms = [f for f in listdir(input_folder_path) if isfile(join(input_folder_path,f)) if f.endswith(".dcm")]
all_dcms = [pydicom.read_file(input_folder_path + '/' + f) for f in listdir(input_folder_path) if isfile(join(input_folder_path,f)) if f.endswith(".dcm")]

all_dcms.sort(key = lambda x: int(x[0x20, 0x32][1]))

all_depth = [x[0x20, 0x32][1] for x in all_dcms]
print(max(all_depth), min(all_depth))
min_depth = -200
max_depth = 200
normalized_depth = [(x - min_depth)/(max_depth - min_depth)*255 for x in all_depth]
print(max(normalized_depth), min(normalized_depth))
#print(all_depth)

dcm_count = 0

if len(all_dcms) == 0:
	print ("No dcm files found")
	exit()

# print(" [ Found " + str(len(all_dcms)) + " images... ]")
# print(" [ Started converting... ]")
# print(' ', end='')
# print('[', end='')

curr_progress = 0
prev_progress = 0

for (dcm, depth) in zip(all_dcms, normalized_depth):

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
    
    image_r = image_2d_scaled[:, :, newaxis]
    image_gb = zeros((512, 512, 2), dtype=image_r.dtype)
    image_rgb = np.c_[image_r, image_gb]
    image_rgb[:, :, 1] = depth
    image_rgb = np.reshape(image_rgb, (512, 1536))
    #print(image_2d[31][31])
    #print(image_rgb[31][30*3 + 1])

    output_name = output_folder_path + "/" + str(dcm_count) + ".png"

    # Write the PNG file
    with open(output_name, 'wb') as png_file:
        w = png.Writer(shape[1], shape[0] , greyscale=False)
        w.write(png_file, image_rgb)

    dcm_count += 1

    if curr_progress - prev_progress > 0.02:
        # print('=', end='')
        prev_progress = curr_progress

print ("\n [ Done! Converted "+ str(dcm_count) + " images ]")
