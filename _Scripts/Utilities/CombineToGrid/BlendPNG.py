##
## Author: Wanze (Russell) Xie
##
## To use this script, you must put the images to combine in the "ToConvert" folder
## You must also create a "Result" folder, where all the output images will be at
##

import cv2
import numpy as np
import copy
import os
from os import walk
import re as reg

input_folder_path = "./ToConvert"
output_folder_path = "./Result"

input_folder_path_raw = "./ToConvert/raw"
input_folder_path_mask = "./ToConvert/mask"

isImageFromTrainingData = 0
isMRIMaskPNGFileNameTheSame = 0
isDrawProgress = 1

'''
This function used to resize images wihout distortion

Usage: image = image_resize(image, height = 512)
And then the width will adjust itself to fit with the height according to the 
original aspect ratio
'''
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA, force_distortion = 0):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    if force_distortion == 1:
    	dim = (width, height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def recolor_mask(input_image, color = [50,205,50]):

	image = copy.deepcopy(input_image)

	white = [255,255,255]
	black = [0,0,0]
	height, width, channels = image.shape

	for x in range(0,width):
		for y in range(0, height):
			channels_xy = image[y,x]
			if all(channels_xy != black):
				image[y,x] = color

	return image

'''
To combine images generated by machines
'''
def load_predicted_images_from_folder():

    raw_images = []
    mask_images = []
    
    fnames = []
    
    # reading and sorting file names
    for (dirpath, dirname, filename) in walk(input_folder_path_raw):
        fnames.extend(filename)

        numbers = []
        fname_to_number = {}

        to_delete = []
        for fname in fnames:
            if(len(reg.findall(r'\d+', fname)) > 0):
                numbers.append(reg.findall(r'\d+', fname)[0])
                fname_to_number[fname] = reg.findall(r'\d+', fname)[0]
            else:
                print("Found weird file and ignored: " + str(fname))
                to_delete.append(fname)

        for ddd in to_delete:
            del fnames[fnames.index(ddd)]

        # fname_to_number = dict(zip(fnames,numbers))
        fnames.sort(key = lambda fname : int(fname_to_number[fname]))

    for filename in fnames:
        img = cv2.imread(os.path.join(input_folder_path_raw,filename))
        if img is not None:
            raw_images.append(img)

            number, file_extension = os.path.splitext(filename)
            mask_filename = number + "_predict.png"
            if isMRIMaskPNGFileNameTheSame == 1:
                mask_filename = filename
            print (mask_filename)
            img_mask = cv2.imread(os.path.join(input_folder_path_mask,mask_filename))

            if img_mask is not None:
                mask_images.append(img_mask)

    return raw_images, mask_images


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


if isImageFromTrainingData == 1:
	raw_images = load_images_from_folder(input_folder_path_raw)
	mask_images = load_images_from_folder(input_folder_path_mask)
else:
    raw_images, mask_images = load_predicted_images_from_folder()

# print (str(len(mask_images)) + " number of mask images")

converted_count = 0
curr_progress = 0
prev_progress = 0

print (len(raw_images))
print (len(mask_images))

for index in range(len(raw_images)):

	curr_progress = converted_count * 1.0 / len(raw_images)
	output_name = output_folder_path + "/" + str(converted_count) + ".png"

	raw = raw_images[index]
	mask = mask_images[index]

	raw = image_resize(raw, width = 512, height = 512, force_distortion = 1)
	mask = image_resize(mask, width = 512, height = 512, force_distortion = 1)

	mask_recolored = recolor_mask(mask)
	blend = cv2.addWeighted(raw,0.8,mask_recolored,0.2,0)
	cv2.imwrite(output_name, blend)

	converted_count += 1

	if isDrawProgress == 1:
		print(str(converted_count)+" ")
		prev_progress = curr_progress





