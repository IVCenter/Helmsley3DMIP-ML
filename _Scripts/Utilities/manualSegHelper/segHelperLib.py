from distutils.core import setup

'''
setup(name='Distutils',
      version='1.0',
      description='Python Distribution Utilities',
      author='Greg Ward',
      author_email='gward@python.net',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['pydicom', 'scikit-image'],
     )
'''

import numpy as np
import png
import pydicom
import os
from os import listdir
from os.path import isfile, join
import shutil
import math
from trainUnet import trainUnet
from testUnet import testUnet
from PIL import Image 
import glob
import re
from sys import exit
from color_dict import *

input_folder_path = "./toSegment"
sampled_dicom_folder_path = "./sampledDataset"
user_segmentation_folder_path = "./sampledDataset"
sampled_label_folder_path = "./sampledLabel"
systemTempPath = "./tmp"

def sampleDicomData (sampleRate:float, dicomPath:str=input_folder_path, outputPath:str=sampled_dicom_folder_path):
    all_dcms = [pydicom.read_file(dicomPath + '/' + f, force=True) for f in listdir(dicomPath) if isfile(join(dicomPath,f)) if f.endswith(".dcm")]
    all_dcms.sort(key = lambda x: int(x[0x20, 0x32][1]))
    dcm_count = 0

    if len(all_dcms) == 0:
        print ("No dcm files found in", dicomPath, ". Abort program.")
        exit()
    
    if(sampleRate <= 0 or sampleRate > 1):
        print("Sample rate must between 0 and 1. Abort program.")
    try:
        os.mkdir(outputPath)
    except OSError:
        print ("Creation of the directory %s failed" % outputPath)
    else:
        print ("Successfully created the directory %s " % outputPath)
    
    sampleStep = math.floor(1/sampleRate)
    sampledData = all_dcms[::sampleStep]
    print("Sampling", len(sampledData), "dicom images.")
    name_counter = 0
    
    for dicom in sampledData:
        pydicom.filewriter.dcmwrite(outputPath + '/' + str(name_counter)+'.dcm', dicom)
        name_counter += 1

def samplePngData (sampleRate:float, pngPath:str, outputPath:str):
    path_list = [im_path for im_path in glob.glob(pngPath)]
    path_list_parsed = [re.split('\\\\|\.', path) for path in path_list]
    path_list_parsed_valid = [x for x in path_list_parsed if x[-1] == 'png']
    path_list_parsed_valid = sorted(path_list_parsed_valid, key=lambda x:int(x[-2]))
    all_pngs = []

    for path in path_list_parsed_valid:
        s = "\\"
        path = [f for f in path if f != '']
        s = s.join(path)
        s = s[:-4] + '.png'
        imageArray = np.array(Image.open(s))
        all_pngs.append(imageArray)

    if len(all_pngs) == 0:
        print ("No dcm files found in", pngPath, ". Abort program.")
        exit()
    
    if(sampleRate <= 0 or sampleRate > 1):
        print("Sample rate must between 0 and 1. Abort program.")
    try:
        os.mkdir(outputPath)
    except OSError:
        print ("Creation of the directory %s failed" % outputPath)
    else:
        print ("Successfully created the directory %s " % outputPath)
    
    sampleStep = math.floor(1/sampleRate)
    sampledData = all_pngs[::sampleStep]
    print("Sampling", len(sampledData), "png images.")
    name_counter = 0
    
    for pngData in sampledData:
        img = Image.fromarray(pngData, 'RGB')
        img.save(outputPath + '/' + str(name_counter) + '.png')
        name_counter += 1

def createTrainingData (imageDicomPath:str=sampled_dicom_folder_path, maskDicomPath:str=sampled_label_folder_path, usePng:bool=False):
    print("Creating training PNG files.")
    systemTempPath = "./tmp"
    
    try:
        os.mkdir(systemTempPath)
    except OSError:
        print ("Creation of the directory %s failed" % systemTempPath)
    else:
        print ("Successfully created the directory %s " % systemTempPath)

    trainingImagePath = "./tmp/trainingImage"
    
    try:
        os.mkdir(trainingImagePath)
    except OSError:
        print ("Creation of the directory %s failed" % trainingImagePath)
    else:
        print ("Successfully created the directory %s " % trainingImagePath)
        
    trainingMaskPath = "./tmp/trainingMask"
    
    try:
        os.mkdir(trainingMaskPath)
    except OSError:
        print ("Creation of the directory %s failed" % trainingMaskPath)
    else:
        print ("Successfully created the directory %s " % trainingMaskPath)
    
    image_dcms = [pydicom.read_file(imageDicomPath + '/' + f) \
                  for f in listdir(imageDicomPath) if isfile(join(imageDicomPath,f)) if f.endswith(".dcm")]
    image_dcms.sort(key = lambda x: int(x[0x20, 0x32][1]))
    
    if(usePng):
        path_list = [im_path for im_path in glob.glob(maskDicomPath)]
        path_list_parsed = [re.split('\\\\|\.', path) for path in path_list]
        path_list_parsed_valid = [x for x in path_list_parsed if x[-1] == 'png']
        path_list_parsed_valid = sorted(path_list_parsed_valid, key=lambda x:int(x[-2]))
        all_masks = []

        for path in path_list_parsed_valid:
            s = "\\"
            path = [f for f in path if f != '']
            s = s.join(path)
            s = s[:-4] + '.png'
            imageArray = np.array(Image.open(s))
            all_masks.append(imageArray)
    else:
        all_masks = [pydicom.read_file(maskDicomPath + '/' + f) \
                  for f in listdir(maskDicomPath) if isfile(join(maskDicomPath,f)) if f.endswith(".dcm")]
        all_masks.sort(key = lambda x: int(x[0x20, 0x32][1]))
    
    if(len(image_dcms) != len(all_masks)):
        print("Error! The number of slices of the image dicoms does not agree with that of the mask dicoms. Abort!")
        exit()
    
    dcm_count = 0

    for image, mask in zip(image_dcms, all_masks):
        shape_image = image.pixel_array.shape
        # Convert to float to avoid overflow or underflow losses.
        image_image_2d = image.pixel_array.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_image_2d,0) / image_image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        
        image_output_name = trainingImagePath + "/" + str(dcm_count) + ".png"
        # Write the PNG file
        with open(image_output_name, 'wb') as png_file:
            w = png.Writer(shape_image[1], shape_image[0], greyscale=True)
            w.write(png_file, image_2d_scaled)

        if(not usePng):
            shape_mask = mask.pixel_array.shape
        
            if(usePng or shape_image != shape_mask):
                print("One or more dicom(s) of the image and the mask do(es) not agree. Abort!")
                exit()
            mask_image_2d = mask.pixel_array.astype(float)
            mask_2d_scaled = (np.maximum(mask_image_2d,0) / mask_image_2d.max()) * 255.0
            mask_2d_scaled = np.uint8(mask_2d_scaled)
            mask_output_name = trainingMaskPath + "/" + str(dcm_count) + ".png"

            with open(mask_output_name, 'wb') as png_file:
                w = png.Writer(shape_mask[1], shape_mask[0], greyscale=True)
                w.write(png_file, mask_2d_scaled)
        else:
            shape_mask = mask.shape
            img = Image.fromarray(mask, 'RGB')
            
            mask_output_name = trainingMaskPath + "/" + str(dcm_count) + ".png"
            img.save(mask_output_name)

        dcm_count += 1

    print ("\n Done! Converted "+ str(dcm_count) + " images and masks.")

def createTestingData(imageDicomPath:str=input_folder_path):
    print("Creating testing PNG files.")
    testingImagePath = "./tmp/testingImage"
    
    try:
        os.mkdir(testingImagePath)
    except OSError:
        print ("Creation of the directory %s failed" % testingImagePath)
    else:
        print ("Successfully created the directory %s " % testingImagePath)
    
    image_dcms = [pydicom.read_file(imageDicomPath + '/' + f) \
                  for f in listdir(imageDicomPath) if isfile(join(imageDicomPath,f)) if f.endswith(".dcm")]
    image_dcms.sort(key = lambda x: int(x[0x20, 0x32][1]))
    
    dcm_count = 0

    for image in image_dcms:
        shape_image = image.pixel_array.shape

        # Convert to float to avoid overflow or underflow losses.
        image_image_2d = image.pixel_array.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_image_2d,0) / image_image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        
        image_output_name = testingImagePath + "/" + str(dcm_count) + ".png"

        # Write the PNG file
        with open(image_output_name, 'wb') as png_file:
            w = png.Writer(shape_image[1], shape_image[0], greyscale=True)
            w.write(png_file, image_2d_scaled)

        dcm_count += 1

    print ("\n Done! Converted "+ str(dcm_count) + " images.")

def createTestingData(imageDicomPath:str=input_folder_path):
    print("Creating testing PNG files.")
    testingImagePath = "./tmp/testingImage"
    
    try:
        os.mkdir(testingImagePath)
    except OSError:
        print ("Creation of the directory %s failed" % testingImagePath)
    else:
        print ("Successfully created the directory %s " % testingImagePath)
    
    image_dcms = [pydicom.read_file(imageDicomPath + '/' + f) \
                  for f in listdir(imageDicomPath) if isfile(join(imageDicomPath,f)) if f.endswith(".dcm")]
    image_dcms.sort(key = lambda x: int(x[0x20, 0x32][1]))
    
    dcm_count = 0

    for image in image_dcms:
        shape_image = image.pixel_array.shape

        # Convert to float to avoid overflow or underflow losses.
        image_image_2d = image.pixel_array.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_image_2d,0) / image_image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        
        image_output_name = testingImagePath + "/" + str(dcm_count) + ".png"

        # Write the PNG file
        with open(image_output_name, 'wb') as png_file:
            w = png.Writer(shape_image[1], shape_image[0], greyscale=True)
            w.write(png_file, image_2d_scaled)

        dcm_count += 1

    print ("\n Done! Converted "+ str(dcm_count) + " images.")

def pruneSystem():

    print("Deleting old training files and temp files.")
    try:
        shutil.rmtree(sampled_dicom_folder_path)
    except OSError:
        print ("Deleting directory %s failed" % sampled_dicom_folder_path)

    try:
        shutil.rmtree(sampled_label_folder_path)
    except OSError:
        print ("Deleting directory %s failed" % sampled_label_folder_path)

    try:
        shutil.rmtree(systemTempPath)
    except OSError:
        print ("Deleting directory %s failed" % systemTempPath)