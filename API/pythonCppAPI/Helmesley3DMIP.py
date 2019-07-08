#python kernel for machine learning result
import sys
import os
import numpy as np
import png
import pydicom
from enum import Enum
from os import listdir
from os.path import isfile, join
from model import *
from data import *

from centerline_module import *
from denoise import *
from tangentPlaneModule import *

class MLStatus(Enum):
	IDLE = 0
	IN_PROGRESS = 1
	ERROR = -1

def DebugLog(output_string):
	print ("(DebugLog): " + output_string)

def DicomConversion(input_folder_path, output_folder_path, verbose = 0):

	all_dcms = [pydicom.read_file(input_folder_path + '/' + f) for f in listdir(input_folder_path) if isfile(join(input_folder_path,f)) if f.endswith(".dcm")]
	all_dcms.sort(key = lambda x: int(x[0x20, 0x32][1]))

	dcm_count = 0

	if len(all_dcms) == 0:
		DebugLog ("Error: No dcm files found")
		exit()

	if (verbose):
		DebugLog ("Started converting dicom to PNGs ...")

	for dcm in all_dcms:
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

	image_count = dcm_count

	if (verbose):
		DebugLog (" [ Done! Converted "+ str(dcm_count) + " images ]")
		DebugLog ("converted PNGs saved to " + output_folder_path)
		DebugLog ("Prediction started...")

	number_of_dicoms = len(all_dcms)

	return number_of_dicoms

def MaskPrediction(input_folder_name, hdf5_file_path, number_of_png, verbose=1):

	path_to_the_mri_images = input_folder_name
	path_to_the_model = hdf5_file_path
	number_of_images = number_of_png

	testGene = testGenerator(path_to_the_mri_images, num_image=number_of_images)
	model = unet()
	model.load_weights(path_to_the_model)
	results = model.predict_generator(testGene,number_of_images,verbose=verbose)

	return results

class MLObject:
	
	def __init__(self):
		self.input_folder_path = "MRI_Images/"
		self.hdf5_file_path = "unet_colon_25_new_2.hdf5"
		self.png_masks_folder = "png_masks_output/"

		self.status = MLStatus.IDLE
		self.image_count = 0
		self.mask_created = False
		self.centerline_file = ""

	def GetPredictedMasksFromDicom(self, path_to_dicom = "", path_to_hdf5 = "", path_to_output = "", verbose = 1):
		self.status = MLStatus.IN_PROGRESS

		is_using_default_path = False;

		if path_to_dicom == "" or path_to_hdf5 == "":
			is_using_default_path = True
			if (verbose):
				DebugLog("Running test using default hdf5 and folder paths...")
		else:

			if path_to_output == "":
				self.input_folder_path = path_to_dicom
				self.hdf5_file_path = path_to_hdf5
			else:
				self.input_folder_path = path_to_dicom
				self.png_masks_folder = path_to_output
				self.hdf5_file_path = path_to_hdf5

				if (verbose):
					DebugLog("PNG masks output folder path was set to be: " + path_to_output)

		temp_folder_name = "temp_ml"
		comm = "mkdir " + temp_folder_name
		os.system(comm)
		
		comm = "rmdir /s/q " + self.png_masks_folder
		os.system(comm)

		comm = "mkdir " + self.png_masks_folder
		os.system(comm)

		number_of_images = DicomConversion(self.input_folder_path, temp_folder_name, verbose=verbose)
		results = MaskPrediction(temp_folder_name, self.hdf5_file_path, number_of_images, verbose=verbose)

		saveResult(self.png_masks_folder,results) 
		
		if (verbose):
			DebugLog ("Cleaning temp folders")

		comm = "rmdir /s/q " + temp_folder_name
		os.system(comm)

		# Denoise step
		results = DenoiseImages(self.png_masks_folder, self.png_masks_folder)

		if (verbose):
			DebugLog ("\n\tCompleted!")

		self.mask_created = True
		self.status = MLStatus.IDLE
	
		return 1

	def GetCenterlineCoords(self, path_to_png_masks = "", verbose = 1):

		self.status = MLStatus.IN_PROGRESS

		if path_to_png_masks == "":
			if self.mask_created == False:
				DebugLog("\n  Error: Trying to use default or preset PNG masks but the masks have not yet been \
					\n         created. If you believe you already have PNG masks in the default or preset folder, \
					\n         you can set \'ignore_valid_check = True\' to force run it. Otherwise, please call \
					\n         the \'GetPredictedMasksFromDicom\' function first to create masks.\n")
				return None
			else:
				if (verbose):
					DebugLog("Using preset png masks path " + self.png_masks_folder)

		else:
			self.png_masks_folder = path_to_png_masks
			if (verbose):
				DebugLog("PNG masks output folder path was set to be: " + path_to_png_masks)

		(samplePointsCorInorder, save_file_name) = GenerateCenterlineCoordinates(self.png_masks_folder, verbose=verbose)

		self.centerline_file_path = save_file_name

		self.status = MLStatus.IDLE

		samplePointsCorInorder_converted = []

		#for center in samplePointsCorInorder:
		#	samplePointsCorInorder_converted.append(center.tolist())

		return (samplePointsCorInorder, save_file_name)


	#TODO: add functions for returning UV coords and distinguish if the 
	#      user is trying to use already made centerline or want to create 
	#      centerline on the fly.
	def GetCuttingPlaneUVLists(self, path_to_centerline_coord = "", segmentPtsDensity = 3, fwdLookLimit = 150):
		
		self.status = MLStatus.IN_PROGRESS
		
		npy_path = ""

		if path_to_centerline_coord == "":
			if self.centerline_file_path == "":
				DebugLog("\n  Error: No centerline path specified. And no default centerline file detected. Aborted")
				return None
			else:
				npy_path = self.centerline_file_path
		else:
			npy_path = path_to_centerline_coord

		self.status = MLStatus.IDLE
		
		return cFuncGetCenterUVFromPtsNpy(npy_path, segmentPtsDensity, fwdLookLimit)
			







				
