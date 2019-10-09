
import numpy as np
import png
import pydicom
import os
from os import listdir, path
from os.path import isfile, join
import shutil
import math
from trainUnet import trainUnet
from testUnet import testUnet
from PIL import Image 
import glob
import re
from sys import exit
from collections import defaultdict
from color_dict import *

input_folder_path = "./toSegment"
sampled_dicom_folder_path = "./sampledDataset"
user_segmentation_folder_path = "./sampledDataset"
sampled_label_folder_path = "./sampledLabel"
systemTempPath = "./tmp"

def getDataPaths():

    dicomDataPaths = defaultdict(list)
    maskDataPaths = defaultdict(list)

    for root, dirs, files in os.walk(".\Dataset\\training", topdown=True):

        for name in files:
            pre, dataId = path.split(root)
            pre, category = path.split(pre)

            if(category == 'image'):
                dicomDataPaths[dataId].append(path.join(root, name))
            elif(category == 'mask'):
                maskDataPaths[dataId].append(path.join(root, name))
    
    return dicomDataPaths, maskDataPaths

def getDataArray(dicomPaths:dict, maskPaths:dict, sortingDirection:int):
    
    if(set(dicomPaths.keys()) != set(maskPaths.keys())):
        print("Error! The series IDs between dicom paths and mask paths don't match.")
        exit()
    
    dicomVolumes = defaultdict(list)
    maskVolumes = defaultdict(list)
    
    for dataId in dicomPaths:
        # Sort all the dicom fules according to the patient position and store them in an array
        image_dcms = [pydicom.read_file(f, force=True) for f in dicomPaths[dataId] if f.endswith(".dcm")]
        image_dcms.sort(key = lambda x: int(x[0x20, 0x32][sortingDirection]))
        
        # Sort all the pngs according their names and store them in an array
        png_path_list = maskPaths[dataId]
        path_list_pre_parsed = [path.split(p) for p in png_path_list]
        path_list_parsed = [ [p[0]] + p[1].split(".") for p in path_list_pre_parsed]
        path_list_parsed_valid = [x for x in path_list_parsed if x[-1] == 'png']
        path_list_parsed_sorted = sorted(path_list_parsed_valid, key=lambda x:int(x[-2]))
        path_list_pre_joined = [ [p[0]] + [(p[1] + (".png"))] for p in path_list_parsed_sorted]
        path_list_joined = [path.join(p[0], p[1]) for p in path_list_pre_joined]
        mask_pngs = [np.array(Image.open(s)) for s in path_list_joined]
        
        if(len(image_dcms) != len(mask_pngs)):
            print("Error! The number of slices of the image dicoms does not agree with that of the mask dicoms. Abort!")
            exit()
            
        dicomVolumes[dataId] = image_dcms
        maskVolumes[dataId] = mask_pngs
    
    return dicomVolumes, maskVolumes

def createTrainingData(dicomPaths:dict, maskPaths:dict, sortingDirection:int, tileN:int=1):
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
    
    dicomVol, maskVol = getDataArray(dicomPaths, maskPaths, sortingDirection)
    slice_count = 0

    for dataId in dicomVol:
        print("Processing data series", dataId)

        black_mask = 0
        white_mask = 0
        
        for image, mask in zip(dicomVol[dataId], maskVol[dataId]):
            image.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            shape_image = image.pixel_array.shape
            
            # Convert to float to avoid overflow or underflow losses.
            image_image_2d = image.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_image_2d,0) / image_image_2d.max()) * 255.0

            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)

            tileSmri = int(image_2d_scaled.shape[0]/tileN)
            tileSmask = int(mask.shape[0]/tileN)
        
            for i in range(tileN):
                for j in range(tileN):
                    mri_tile = image_2d_scaled[i*tileSmri:(i+1)*tileSmri, j*tileSmri:(j+1)*tileSmri]
                    mask_tile = mask[i*tileSmask:(i+1)*tileSmask, j*tileSmask:(j+1)*tileSmask]
                    
                    if (np.any(mask_tile)):
                        white_mask += 1
                    else:
                        black_mask += 1
                    

                    image_output_name = trainingImagePath + "/" + format(slice_count, '05d') + ".png"
                    # Write the PNG file
                    mri_img = Image.fromarray(mri_tile, 'L')
                    mri_img.save(image_output_name)

                    shape_mask = mask.shape
                    mask_img = Image.fromarray(mask_tile)
                    mask_output_name = trainingMaskPath + "/" + format(slice_count, '05d') + ".png"
                    mask_img.save(mask_output_name)

                    slice_count += 1
            
        print(black_mask, white_mask)
            
            

def createTestingData(sortingDirection:int, testSeriesName:str, imageDicomBasePath:str="./Dataset/testing/", tileN:int=1):
    imageDicomPath = imageDicomBasePath + testSeriesName
    
    print("Creating testing PNG files.")
    testingImagePath = "./tmp/testingImage"
    
    try:
        os.mkdir(testingImagePath)
    except OSError:
        print ("Creation of the directory %s failed" % testingImagePath)
    else:
        print ("Successfully created the directory %s " % testingImagePath)
    
    image_dcms = [pydicom.read_file(imageDicomPath + '/' + f, force=True) \
                  for f in listdir(imageDicomPath) if isfile(join(imageDicomPath,f)) if f.endswith(".dcm")]
    image_dcms.sort(key = lambda x: int(x[0x20, 0x32][sortingDirection]))
    
    slice_count = 0

    for image in image_dcms:

        # In case the dicom file does not contain certain meta tags
        image.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        shape_image = image.pixel_array.shape

        # Convert to float to avoid overflow or underflow losses.
        image_image_2d = image.pixel_array.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_image_2d,0) / image_image_2d.max()) * 255.0

        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        
        tileSmri = int(image_2d_scaled.shape[0]/tileN)

        for i in range(tileN):
            for j in range(tileN):

                mri_tile = image_2d_scaled[i*tileSmri:(i+1)*tileSmri, j*tileSmri:(j+1)*tileSmri]
                image_output_name = testingImagePath + "/" + format(slice_count, '05d') + ".png"

                # Write the PNG file
                mri_img = Image.fromarray(mri_tile, 'L')
                mri_img.save(image_output_name)

                slice_count += 1

    print ("\n Done! Converted "+ str(slice_count) + " images.")

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

def combineTiles(inputpath:str, tileN:int):
    testImagePath = "./testCombine"
    png_path_list = []

    for root, dirs, files in os.walk(inputpath, topdown=True):

        for name in files:
            png_path_list.append(path.join(root, name))
        
        # Sort all the pngs according their names and store them in an array
        path_list_pre_parsed = [path.split(p) for p in png_path_list]
        path_list_parsed = [ [p[0]] + p[1].split(".") for p in path_list_pre_parsed]
        path_list_parsed_valid = [x for x in path_list_parsed if x[-1] == 'png']
        path_list_parsed_sorted = sorted(path_list_parsed_valid, key=lambda x:int(x[-2]))
        path_list_pre_joined = [ [p[0]] + [(p[1] + (".png"))] for p in path_list_parsed_sorted]
        path_list_joined = [path.join(p[0], p[1]) for p in path_list_pre_joined]
        tile_pngs = [np.array(Image.open(s)) for s in path_list_joined]
        print(tile_pngs[0].shape)
        tile_length = tile_pngs[0].shape[0]

        output_shape = (tile_length*4, tile_length*4, 3)
        sliceCount = 0

        for count in range(int(len(tile_pngs)/(tileN**2))):
            
            mask_tiles = tile_pngs[count*tileN**2:(count+1)*tileN**2]
            outputImage = np.zeros(output_shape)

            for i in range(tileN):
                for j in range(tileN):
                    outputImage[i*tile_length:(i+1)*tile_length, j*tile_length:(j+1)*tile_length] = mask_tiles[i * tileN + j]

            image_output_name = testImagePath + "/" + format(sliceCount, '05d') + ".png"
            print(outputImage.shape)
            # Write the PNG file
            mri_img = Image.fromarray(outputImage.astype('uint8'))
            mri_img.save(image_output_name)
            sliceCount += 1


            
