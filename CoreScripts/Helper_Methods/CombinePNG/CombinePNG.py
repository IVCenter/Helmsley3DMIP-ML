##
## Author: Wanze (Russell) Xie
##

import cv2
import numpy as np
import copy
import os


input_folder_path = "./ToConvert"
output_folder_path = "./Result"

input_folder_path_raw = "./ToConvert/raw"
input_folder_path_mask = "./ToConvert/mask"

isImageFromTrainingData = 1
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
def load_predicted_images_from_folder(folder):

    raw_images = []
    mask_images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
        	if "predict" in filename:
        		mask_images.append(img)
        	else:
        		raw_images.append(img)

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
	raw_images, mask_images = load_predicted_images_from_folder(input_folder_path)


converted_count = 0
curr_progress = 0
prev_progress = 0


for index in range(len(raw_images)):

	curr_progress = converted_count * 1.0 / len(raw_images)
	output_name = output_folder_path + "/" + str(converted_count) + ".png"

	raw = raw_images[index]
	mask = mask_images[index]

	raw = image_resize(raw, width = 512, height = 512, force_distortion = 1)
	mask = image_resize(mask, width = 512, height = 512, force_distortion = 1)
	vis = np.concatenate((raw, mask), axis=1)

	mask_recolored = recolor_mask(mask)
	blend = cv2.addWeighted(raw,0.8,mask_recolored,0.2,0)
	addition = cv2.add(raw,mask)
	vis2 = np.concatenate((blend,addition), axis=1)

	res = np.concatenate((vis,vis2),axis=0)
	cv2.imwrite(output_name, res)

	converted_count += 1

	if isDrawProgress == 1:
		print(str(converted_count)+" ", end='')
		prev_progress = curr_progress





