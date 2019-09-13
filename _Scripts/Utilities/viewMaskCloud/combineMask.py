import numpy as np
from color_dict import *
from PIL import Image
import sys

if(len(sys.argv) == 2):
    try:
        img_num = int(sys.argv[1])
    except:
        print(sys.argv[1], " is not an integer. Abort.")
        sys.exit(0)
else:
    print("This function takes exactly one argument (number of images for each kind of organs). Abort.")
    sys.exit(0)

#whole program loops through each 3 sets of images per folder. 
#8 bit encoding key for more variance
kidneyMaskPath = "./Kidney"
bladderMaskPath = "./Bladder"
colonMaskPath = "./Colon"
combinedOutputPath = "./Combined"

i = 0
for i in range(img_num):
    kidneyImg = np.array(Image.open(kidneyMaskPath + '/' + str(i) + '.png'))
    kidneyLoc = np.where(kidneyImg > 0) 
    colonImg = np.array(Image.open(bladderMaskPath + '/' + str(i) + '.png'))
    colonLoc = np.where(colonImg > 0)
    bladderImg = np.array(Image.open(colonMaskPath + '/' + str(i) + '.png'))
    bladderLoc = np.where(bladderImg > 0)
    
    if(kidneyImg.shape == colonImg.shape == bladderImg.shape):
        imageShape = kidneyImg.shape
    else:
        print("Error! One or more image shape of the masks don't match!")
    imgArray = np.zeros((512, 512, 3))
    imgArray += Unlabelled_color
    
    # Give 
    if(len(colonLoc[0]) > 0):
        imgArray[colonLoc[0], colonLoc[1]] = Colon_color
    
    if(len(kidneyLoc[0]) > 0):
        imgArray[kidneyLoc[0], kidneyLoc[1]] = Kidney_color
        
    if(len(bladderLoc[0]) > 0):
        imgArray[bladderLoc[0], bladderLoc[1]] = Bladder_color
    
    imgArray = imgArray.astype("uint8")
    img = Image.fromarray(imgArray, 'RGB')
    img.save(combinedOutputPath + '/' + str(i) + '.png')
