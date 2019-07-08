from model import *
from data import *


# read over the directory, count the number of files.  
# parse two strings for input out put  
# dicom to mask directly.  
# save as numpy array makes better  
# run python on the same heap-space.  

path_to_the_mri_images = "MRI_Images"
path_to_the_model = "unet_colon.hdf5"
path_to_the_output_folder = "output"
number_of_images = 9

testGene = testGenerator(path_to_the_mri_images)
model = unet()
model.load_weights(path_to_the_model)
results = model.predict_generator(testGene,number_of_images,verbose=1)
saveResult(path_to_the_output_folder,results) 