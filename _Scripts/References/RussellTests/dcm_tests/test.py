##########################
#                        #
#    Just a test file    #
#                        #
##########################


from imebra.imebra import *

import numpy as np
import scipy.misc as smp


loadedDataSet = CodecFactory.load("Z937.dcm")

print (loadedDataSet)
# # A patient's name can contain up to 5 values, representing different interpretations of the same name
# # (e.g. alphabetic representation, ideographic representation and phonetic representation)
# # Here we retrieve the first 2 interpretations (index 0 and 1)
# patientNameCharacter = loadedDataSet.getString(TagId(tagId_t_PatientName_0010_0010), 0)
# patientNameIdeographic = loadedDataSet.getString(TagId(tagId_t_PatientName_0010_0010), 1)

# print patientNameCharacter
# print patientNameIdeographic

# Retrieve the first image (index = 0)
image = loadedDataSet.getImageApplyModalityTransform(0)

# Get the color space
colorSpace = image.getColorSpace()

# Get the size in pixels
width = image.getWidth()
height = image.getHeight()

print (width)
print (height)

# let's assume that we already have the image's size in the variables width and height
# (see previous code snippet)

# Retrieve the data handler
dataHandler = image.getReadingDataHandler()

# Create a 1024x1024x3 array of 8 bit unsigned integers
data = np.zeros( (width,height,3), dtype=np.uint8 )


for scanY in range(0, height):

    for scanX in range(0, width):

        # For monochrome images
        luminance = dataHandler.getSignedLong(scanY * width + scanX)

        # # # For RGB images
        # r = dataHandler.getSignedLong((scanY * width + scanX) * 3)

        # g = dataHandler.getSignedLong((scanY * width + scanX) * 3 + 1)

        # b = dataHandler.getSignedLong((scanY * width + scanX) * 3 + 2)

        data[scanX, scanY] = [luminance,luminance,luminance]

img = smp.toimage( data )       # Create a PIL image
img.show()                      # View in default viewer


