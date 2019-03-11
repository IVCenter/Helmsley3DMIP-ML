from model import *
from data import *
import os
import tempfile
import datetime

import pydicom
from pydicom.dataset import Dataset, FileDataset

def saveResultDicom(save_path,npyfile):
	for i,item in enumerate(npyfile):
		
		suffix = '.dcm'
		filename_little_endian = save_path + str(i) + suffix;

		print("Setting file meta information...")
		# Populate required values for file meta information
		file_meta = Dataset()
		file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
		file_meta.MediaStorageSOPInstanceUID = "1.2.3"
		file_meta.ImplementationClassUID = "1.2.3.4"

		print("Setting dataset values...")
		# Create the FileDataset instance (initially no data elements, but file_meta
		# supplied)
		ds = FileDataset(filename_little_endian, {},
		                 file_meta=file_meta, preamble=b"\0" * 128)

		# Add the data elements -- not trying to set all required here. Check DICOM
		# standard
		ds.PatientName = "2017_test"
		ds.PatientID = "123456"

		# Set the transfer syntax
		ds.is_little_endian = True
		ds.is_implicit_VR = True

		# Set creation date/time
		dt = datetime.datetime.now()
		ds.ContentDate = dt.strftime('%Y%m%d')
		timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
		ds.ContentTime = timeStr
		
		# Set pixel_data
		ds.PixelData = item;
		
		print("Writing test file", filename_little_endian)
		ds.save_as(filename_little_endian)
		print("File saved.")

testGene = testGenerator("data/membrane/2017_imgs")
model = unet()
model.load_weights("unet_model_144_2016.hdf5")
results = model.predict_generator(testGene,5,verbose=1)
saveResultDicom("data/membrane/2017_predict/",results)