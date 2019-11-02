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
RGBMaskPath = "./2016_144"
convertedOutputPath = "./converted"

i = 0

for i in range(img_num):
    rgbMaskImg = np.array(Image.open(RGBMaskPath + '/' + str(i) + '.png'))
    rgbMaskImg = np.stack((rgbMaskImg, rgbMaskImg ,rgbMaskImg), axis=-1)
    imgArray = np.zeros((512, 512, 3))

    for j in range(len(COLOR_DICT)):
        index = np.where(np.all(rgbMaskImg == COLOR_DICT[j], axis = 2))
        
        if(len(index[0]) > 0):
            imgArray[index[0], index[1]] = BIT_DICT[j]
    
    
    imgArray = imgArray.astype("uint8")
    img = Image.fromarray(imgArray, 'RGB')
    img = img.resize((512, 512), 0)
    img.save(convertedOutputPath + '/' + str(i) + '.png')
