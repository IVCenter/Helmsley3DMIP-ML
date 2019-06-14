from os import walk
# for file names retrival
import re as reg
# for sorting file names
from PIL import Image
# for png image reading
import numpy as np
# you know the deal
import matplotlib.pyplot as plt
# for ploting
import pickle
# for python object saving
import networkx as nx
# for graph visualization and searching for master tree
from scipy.misc import imresize
# for filter up sample
from denoise_param import *
# denoise parameters
from itkwidgets import view
import cv2

h_size = 0
numb_layers = 0


'''
This function used to resize images wihout distortion

Usage: image = image_resize(image, height = 512)
And then the width will adjust itself to fit with the height according to the 
original aspect ratio
'''
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA, force_distortion = 0):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    if force_distortion == 1:
    	dim = (width, height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def make_graph(numpy_hist, radius = 3, layer_interval = 1):
    print('making graph based on 3d-histogram')
    def valid_idx(tuple, shape):            
        if 0 <= tuple[0] < shape[0] \
        and 0 <= tuple[1] < shape[1] \
        and 0 <= tuple[2] < shape[2]:
            return True
        return False
            
    # num_layer x w x h
    num_layers = numpy_hist.shape[0]
    num_rows = numpy_hist.shape[1]
    num_cols = numpy_hist.shape[2]

    edge_set = []
    vert_set = []

    # create vertices
    for layer in range(num_layers):
        for row in range(num_rows):
            for col in range(num_cols):                
                if numpy_hist[layer][row][col] == 1:
                    vert_set.append((layer,row,col))

    # create edges
    for layer in range(num_layers):
        for row in range(num_rows):
            for col in range(num_cols):              
                current_vert = (layer, row, col)
                # search neightbor and connect them
                for x in range(1,layer_interval+1):
                    for y in range(1,radius+1):
                        for z in range(1,radius+1):
                            # print(z,y,x)
                            neightbor = (layer + z, row + y, col + x)
                            if valid_idx(neightbor, numpy_hist.shape):
                                if numpy_hist[neightbor] == 1:
                                    edge_set.append((current_vert,neightbor))

    return vert_set, edge_set

def save_object(obj, filename):
    print(f'saving file: {filename}')
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    print(f'loading file: {filename}')
    with open(filename, 'rb') as input_file:
        input_obj = pickle.load(input_file)
    return input_obj

def save_img_from_np(numpy_masks, path):
    print(f'saving image to path: {path}')
    for idx, np_mask in enumerate(numpy_masks):
        im = Image.fromarray(np_mask*255)
        im.save(f'{path}/{idx}.png')

def playMaskSequence(numpy_masks):
    img_buff = None
    for np_mask in numpy_masks:
        #print(np_mask.max())
        if img_buff is None:
            img_buff = plt.imshow(np_mask)
        else:
            img_buff.set_data(np_mask)
        plt.pause(.2)
        plt.draw()
    
    plt.show()

def getNumpyMasks(pathname):
    print ("pathname:",pathname)
    # numpy masks to be returned
    numpy_masks = []

    # sorted png file names to be read using pilow
    fnames = []
    
    # pixel threshold, we turn it to 1 if the pixel is larger
    # than this threshold
    pixel_t = 50
    # reading and sorting file names
    for (dirpath, dirname, filename) in walk(pathname):
        fnames.extend(filename)

        numbers = []
        fname_to_number = {}

        to_delete = []
        for fname in fnames:
            if(len(reg.findall(r'\d+', fname)) > 0):
                numbers.append(reg.findall(r'\d+', fname)[0])
                fname_to_number[fname] = reg.findall(r'\d+', fname)[0]
            else:
                print("Found weird file and ignored: " + str(fname))
                to_delete.append(fname)

        for ddd in to_delete:
            del fnames[fnames.index(ddd)]

        # fname_to_number = dict(zip(fnames,numbers))
        fnames.sort(key = lambda fname : int(fname_to_number[fname]))

    # reading and converting png file to one numpy slice
    for filename in fnames:
        picture = Image.open(pathname + "/" + filename)
        np_slice = np.array(picture).astype('uint8')
        np_slice = (np_slice > pixel_t)
        numpy_masks.append(np_slice)
    return np.array(numpy_masks).astype('uint8')

def gen3DHistogram(numpy_masks, texel_threshold = 30):
    print('generating 3d histogram')
    
    global h_size
    global numb_layers
    
    # size for the histogram
    h_size = 64
    # grid size in terms of pixel (as 256 / 32 = 8)
    unit_size = 4
    
    # number of layers for the masks
    numb_layers = numpy_masks.shape[0]

    # histogram holding the result
    histogram = np.zeros((numb_layers, h_size, h_size))

    for layer in range(numb_layers):
        for row in range(h_size):
            for col in range(h_size):
                # within a grid now
                    for pixel_w in range(unit_size):
                        for pixel_h in range(unit_size):
                            # print(f'Before p_w: {pixel_w}, p_h: {pixel_h}')
                            _pixel_w = unit_size * row + pixel_w
                            _pixel_h = unit_size * col + pixel_h
                            '''
                            print(f'p_w: {_pixel_w}, p_h: {_pixel_h}')
                            print(f'row: {row}, col: {col}')
                            '''
                            if numpy_masks[layer][_pixel_w][_pixel_h] > 0:
                                histogram[layer][row][col] = histogram[layer][row][col] + 1

    histogram[np.where(histogram <= texel_threshold)] = 0 
    histogram[np.where(histogram > texel_threshold)] = 1

    print ( "Number of 1s in histogram: " + str(histogram.sum()))

    return histogram.astype('uint8')


def make_filter(master_vert, hist):
    print('making a filter using master tree')
    a_filter = np.zeros((numb_layers, h_size, h_size)).astype('uint8')
    for vert in master_vert:
        a_filter[vert] = 1
    print(a_filter.sum())
    a_filter = a_filter * hist.astype('uint8')
    print(a_filter.sum())
    return a_filter

def upsample_filter(afilter):
    new_filter = []
    for filter_slice in afilter:
        pic = Image.fromarray(filter_slice * 255).resize((256,256))
        filter_slice = np.array(pic)
        filter_slice = filter_slice / 255
        new_filter.append(filter_slice)
    return np.array(new_filter).astype('uint8')
    
# original = getNumpyMasks('./2017_inner')
# save_object(original, 'original')
# original = load_object('original')
# save_object(gen3DHistogram(original, texel_threshold), '3d_hist')
# hist = load_object('3d_hist')
# save_object(make_graph(hist, neighbor_distance, slice_levels),'graph')
# v, e = load_object('graph')

# g = nx.Graph()
# g.add_nodes_from(v)
# g.add_edges_from(e)
# ccs = list(nx.connected_components(g))
# size_cc = 0
# target_cc = ccs[0]
# for cc in ccs:
#     if size_cc <= len(cc):
#         size_cc = len(cc)
#         target_cc = cc


# a_filter = make_filter(target_cc, hist)
# save_object(a_filter, 'filter')

# a_filter = load_object('filter')
# a_filter = upsample_filter(a_filter)

# save_img_from_np(a_filter, './2017_inner_up')
# save_img_from_np(a_filter * getNumpyMasks('./2017_inner'), './2017_inner_post')

# save_object(a_filter * getNumpyMasks('./2017_inner'), 'result')

original = getNumpyMasks('./2017_inner')

hist = gen3DHistogram(original, texel_threshold)
v, e = make_graph(hist, neighbor_distance, slice_levels)

g = nx.Graph()
g.add_nodes_from(v)
g.add_edges_from(e)
ccs = list(nx.connected_components(g))
size_cc = 0
target_cc = ccs[0]
for cc in ccs:
    if size_cc <= len(cc):
        size_cc = len(cc)
        target_cc = cc


a_filter = make_filter(target_cc, hist)
a_filter = upsample_filter(a_filter)

save_img_from_np(a_filter * getNumpyMasks('./2017_inner'), './2017_inner_post')
save_object(a_filter * getNumpyMasks('./2017_inner'), 'result')
