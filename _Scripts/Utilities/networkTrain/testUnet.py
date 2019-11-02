import os
from os import listdir
from os.path import isfile, join
import sys
import time
import datetime
from test_model import *
from data import *
from tensorflow.python.client import device_lib

def testUnet(colorDict, modelPath:str, input_size, trainX):

	print(device_lib.list_local_devices())

	'''
	Get time
	'''
	now = time.time()
	time_stamp = datetime.datetime.fromtimestamp(now).strftime('_%m_%d_%H_%M')

	input_folder = "./tmp/testingImage"

	output_folder = "./predicted_segmnetation" + time_stamp

	num_images = len([f for f in listdir(input_folder) if isfile(join(input_folder,f)) if f.endswith(".png")])

	if(num_images < 1):
		print("There is no testing images generated. Abort!")
		return -1

	'''
	Run the Test scripts
	'''
	testGene = testGeneratorTest(trainX, target_size=input_size)
	model, cpuModel = unet(numLabels=len(colorDict), input_size=input_size)
	model.load_weights(modelPath)

	results = model.predict_generator(testGene,num_images,verbose=1)
	'''
	Save the results
	'''
	try:
		os.mkdir(output_folder)
	except OSError:
		print ("Creation of the directory %s failed" % output_folder)
	else:
		print ("Successfully created the directory %s " % output_folder)

	saveResult(output_folder, results, num_class=len(colorDict), color_dict=colorDict, img_size=input_size[:2])



