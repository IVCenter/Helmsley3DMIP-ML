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
additionalOrganPath = "./Additional"
currentMaskPath = "./Current"
newCombinedPath = "Combined"

i = 0
for i in range(img_num):
    currentImg = np.array(Image.open(currentMaskPath + '/' + str(i) + '.png'))
    additionalImg = np.array(Image.open(additionalOrganPath + '/' + str(i) + '.png'))
    additionalLoc = np.where(additionalImg > 0)
    
    # Give 
    if(len(additionalLoc) > 0):
        currentImg[additionalLoc[0], additionalLoc[1]] = Splean_clolor
    
    currentImg = currentImg.astype("uint8")
    img = Image.fromarray(currentImg, 'RGB')
    img.save(newCombinedPath + '/' + str(i) + '.png')
