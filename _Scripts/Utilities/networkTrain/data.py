from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from sys import exit
from PIL import Image, ImageEnhance 


def adjustData(img,mask,organ_color_dict,sample_step):
    img = img / 255

    if(mask.shape[-1] != 3):
        print("Input mask must have RGB channels.")
        exit()

    # Create an empty place holder fot the adjusted masks. The new mask will use one-hot encodering
    new_mask = np.zeros(mask.shape[:3] + (len(organ_color_dict),))

    # Filling the the new mask. 
    for i in range(len(organ_color_dict)):
        index = np.where(np.all(mask == organ_color_dict[i], axis = 3))
        new_mask[index[0], index[1], index[2], i] = 1
        #sprint("Class", i, ":", len(index[0]))

    mask = new_mask

    return (img[:, ::sample_step, ::sample_step, :],mask[:, ::sample_step, ::sample_step, :])

def contrastAdjust(img):
    enhancer = ImageEnhance.Contrast(img)
    enhancer.enhance(factor)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,organ_color_dict, image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    multi_label = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)

    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask, organ_color_dict)
        yield (img,mask)

def trainGeneratorTest(batch_size, x, y, aug_dict,organ_color_dict,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow(
        x,
        batch_size = batch_size,
        seed = seed)
    mask_generator = mask_datagen.flow(
        y,
        batch_size = batch_size,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)

    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask, organ_color_dict, 2)
        yield (img,mask)

def trainGenerator3D(batch_size, x, y, aug_dict,organ_color_dict,seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow(
        x,
        batch_size = batch_size,
        seed = seed)
    mask_generator = mask_datagen.flow(
        y,
        batch_size = batch_size,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)

    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask, organ_color_dict, 2)
        yield (img,mask)


def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,format(i, '05d') + '.png'),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img

def testGeneratorTest(testX,target_size = (256,256)):
    for i in range(len(testX)):
        img = testX[i]
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img

def labelVisualize(num_class,color_dict,item):
    img_out = np.zeros(item.shape[:2] + (3,))
    threshold = 0.0

    for i in range(num_class):
        # Find the category with the largest possibility of a pixel
        classLoc = np.where(np.argmax(item, axis=2) == i)
        img_out[classLoc[0], classLoc[1]] = color_dict[i]

        confidenceValues = item[classLoc[0], classLoc[1], i]
        unsureConfidenceIndex = np.where(confidenceValues < threshold)
        unsurePtsLoc = [classLoc[0][unsureConfidenceIndex], classLoc[1][unsureConfidenceIndex]]
        #print(values.shape)
        img_out[unsurePtsLoc[0], unsurePtsLoc[1]] = np.array([255, 255, 0])

        #print("Class", i, ":", len(index[0]))

    return img_out



def saveResult(save_path,npyfile,num_class:int, color_dict, img_size):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,color_dict,item) 
        img = img.astype('uint8')
        img = trans.resize(img, img_size)
        io.imsave(os.path.join(save_path,"%d.png"%i),img)



        
