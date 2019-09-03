from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from os import walk
import re as reg

# --------------------------------------------------------------------
# ** IMPORTANT **: Make sure your put your file name below correctly, 
#                  Then run 'python dice.coeff.py'
# --------------------------------------------------------------------
# put your ground truth images folder here
input_folder_path_ground_truth = "./gt"
# put your predicted images folder here
input_folder_path_predicted_labels = "./sample_predict2_denoise"

def load_images_from_folder(folder):

    pathname = folder
    print ("pathname:",pathname)

    # sorted png file names to be read using pilow
    fnames = []

    # reading and sorting file names
    for (dirpath, dirname, filename) in walk(pathname):
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

    images = []
    for filename in fnames:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


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


def dice_coef(img, img2):
    if img.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
    else:
        intersection = np.logical_and(img, img2)
        if img.sum() == 0 and img2.sum() == 0 or img2.sum() < 100 or img.sum() == 0:
            return 1.0
        value = (2. * intersection.sum())  / (img.sum() + img2.sum())

        # if value == 0:
        #     print("value==0: " + str(img.sum()) + ", "+ str(img2.sum()))
    return value 

def make_grey(img):

    if img.shape != (256,256,3):
        raise ValueError("Shape mismatch: RGB Image shape has to be (256,256,3).")

    img = np.delete(img,1,axis=2)
    img = np.delete(img,1,axis=2)
    img = img.reshape(256,256)

    return img


if __name__ == "__main__":

    print ("-" * 40)
    print ("loading files...")
    gt_images = load_images_from_folder(input_folder_path_ground_truth)
    predict_images = load_images_from_folder(input_folder_path_predicted_labels)

    print ("Groundtruth images found: " + str(len(gt_images)))
    print ("Predicted images found: " + str(len(predict_images)))
    print ("-" * 40)
    
    value = 0

    for index in range(len(gt_images)):
        img = gt_images[index]
        img2 = predict_images[index]

        img = image_resize(img, width = 256, height = 256, force_distortion = 1)
        img2 = image_resize(img2, width = 256, height = 256, force_distortion = 1)
        
        if img.shape != (256,256,3):
            print(img.shape)
            print(img2.shape)
        
        # ignore pixel value that is less than 32, sometimes you would get better result, or worse.
        # pixel_threshold = 32      
        
        # img[np.where(img <= pixel_threshold)] = 0 
        # img[np.where(img > pixel_threshold)] = 1
        
        # img2[np.where(img2 <= pixel_threshold)] = 0 
        # img2[np.where(img2 > pixel_threshold)] = 1

        img = np.asarray(img).astype(np.bool)
        img2 = np.asarray(img2).astype(np.bool)

        value_img = dice_coef(img, img2) 

        print("Image #" + str(index) + ", DC=" + str(value_img))

        value += value_img / len(gt_images)

    print ("-" * 40)
    print ("FINAL: DICE COEFF IS: " + str(value))
    print ("-" * 40)

















