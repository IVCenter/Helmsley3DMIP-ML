import numpy as np
import networkx as nx
import imageio
import glob
import re
from random import sample, seed
from scipy.spatial import distance
from time import strftime
import os
import math
import time
import copy
from tempfile import TemporaryFile
from collections import defaultdict
from math import *
import time
import datetime
import progressbar
import warnings

#import pptk

seed(20)
if os.name == 'nt': # Windows
    system_win = 1
else:
    system_win = 0

graph_base = None
pointsCor3D_base = None
removedNodeDict = None
graph_centerline = None
pointsCor3D_centerline = None

'''
Get time
'''
now = time.time()
time_stamp = datetime.datetime.fromtimestamp(now).strftime('_%m_%d_%H_%M')


# Given the data points, return the MST of the point cloud generated from the PNG files and an array 
# of points positions
# Param: data: an array of points coordinate. Each point's coordinate should has the format of [x, y, z]
#        drawMST: boolean value. Default is false. If set true, the function will also draw a MST graph at the end
#        sampleNumber: int value. Default is 5000. This function will only sample <sampleNumber> points from the data
# Return: a NetworkX graph representing the Minimum spanning tree of the data points


# This function will invoke a pptk viewer to render the points 
# Param: data: an array of points coordinate. Each point's coordinate should has the format of [x, y, z]
#       pointSize: the size of the point to be rendered on the screen
# Return: none

#def displayPoints(data, pointSize):
#    v = pptk.viewer(data)
#    v.set(point_size=pointSize)


def getMSTFromDataPoint(data, drawMST: bool=False, sampleNumber: int=5000,  maxNeighborDis:int=10):
    # Read points data from PNGs 
    if(sampleNumber > len(data)):
        sampleNumber = len(data)
        
    # default sample 5000 points from the whole set, otherwise it would take too long
    print("---------------")
    print("There are " + str(len(data)) + " points in total. Sampled " + str(sampleNumber) + " points from them")
    sample_data = np.asarray(sample(list(data), sampleNumber))
    
    #display the points 
    #displayPoints(sample_data, 1.3)
    
    #Create a networkX graph instance that represent MST
    print("---------------")
    print("Creating a MST of the sampled points cloud")
    MST = CreateMSTGraph(sample_data, maxNeighborDis)

    if(drawMST):
        nx.draw(MST, dict(enumerate(sample_data[:, :2])))

    return (MST, sample_data)


# This function will read a txt file and convert its content to point data, which is an array of points coordinate. 
# Each point's coordinate should has the format of [x, y, z]
# Param: filePath: the file path of the txt file
# Return: an array of points coordinate. Each point's coordinate has the format of [x, y, z]

def readPointFromTXT(filePath):
    data = np.genfromtxt(fname=filePath, skip_header=0)
    return data


# This function will read a series of PNG file and convert its content to point data, which is an array of points coordinate. 
# Each point's coordinate will have the format of [x, y, z]
# Param: filePath: the file path of the PNG files. Each file should be named as 1.png, 2.png, 3.png ... etc. All the png file should 
#                  be ordered by the their topological order from their original dicom file
#        orientation: 0, 1, or 2. 0 stands for coronal. 1 stands for transverse. 2 stands for sagittal.
# Return: an array of points coordinate. Each point's coordinate has the format of [x, y, z]

def ReadPointFromPNG(filepath, orientation:int, padding:int):
    path_list = [im_path for im_path in glob.glob(filepath)]
    
    if system_win:
        path_list_parsed = [re.split('\\\\|\.', path) for path in path_list]
    else:
        path_list_parsed = [re.split('/|\.', path) for path in path_list]
    
    path_list_parsed_striped = []
    for path in path_list_parsed:
        path = [x for x in path if x != '']
        path_list_parsed_striped.append(path)
        
    path_list_parsed_valid = [x for x in path_list_parsed_striped if x[-1] == 'png']
    path_list_parsed_valid = sorted(path_list_parsed_valid, key=lambda x:int(x[-2]))
    
    print("There are", len(path_list_parsed_valid),"PNG files, now converting them to point coordinates in 3D")
    imageData = []
    
    for path in path_list_parsed_valid:
        s = ""
        if system_win:
            s = "\\"
        else:
            s = "/"
        s = s.join(path)
        s = s[:-4] + '.png'
        image = imageio.imread(s)
        
        for i in range(padding):
            imageData.append(image)
   
    # Transfrom the matrix to list of points' coordinate whose grey scalse is not 0 (colon area)
    if(orientation == 0):
        zxy = np.transpose(np.where(imageData))
        xyz = zxy[:, [1, 2, 0]]
        #xyz[:, 2] = xyz[:, 2]*3*thickness
        
    elif(orientation == 1):
        yxz = np.transpose(np.where(imageData))
        xyz = yxz[:, [1, 0, 2]]
        #xyz[:, 0] = xyz[:, 0]*3*thickness
        
    elif(orientation == 2):
        zxy = np.transpose(np.where(imageData))
        xyz = zxy[:, [0, 1, 2]]
        #xyz[:, 0] = xyz[:, 0]*3*thickness
    
    else:
        print("Orientation shoud only be one of 0, 1 or 2 only.  0 stands for coronal. \
        1 stands for transverse. 2 stands for sagittal.")
        
    return xyz

# This function is used to limited the number of edges in the original graph.
# Instead of creating a graph with full connectivity, this function will return 
# a list of neighbor points for each point and we will only connect them in the graph
# Param: pointsData: an array of points coordinate. Each point's coordinate has the format of [x, y, z]
# return: a tuple(closestIndices, closesDis). ClosestIndices is a matrix of each point's neighbors. 
#         closestDis is a matrix of the distances between each point and their neighbors

def getNearbyPoints(pointsData, maxNeighborDis):
    D = distance.squareform(distance.pdist(pointsData))
    closestIndicies = np.argsort(D, axis=1)
    closestDis = np.sort(D, 1)
    threshold = maxNeighborDis # This number can be changed. The greater this number, the more edges
    return (closestIndicies[:, 1:threshold], closestDis[:, 1:threshold])



# This function converts points' coordinate data into a minimum spanning tree. In this graph, the nodes are the points
# from the points cloud and the edges are the connection between each point and their neighbors. The weights are each 
# connection's distance in space
# Param: pointsData: an array of points coordinate. Each point's coordinate has the format of [x, y, z]
# Return: A networkX instance containing the MST

def CreateMSTGraph(pointsData, maxNeighborDis):

    # The following variables are used to store the distances between points

    print("---------------")
    print("Begin calculating nearby points for each point")
    nearbyInfo = getNearbyPoints(pointsData, maxNeighborDis)
    closestIndicies = nearbyInfo[0]
    closestDis = nearbyInfo[1]
    print("---------------")
    print("Nearby points calculation Done! Total:", closestIndicies.shape[0], "points,",\
          closestIndicies.shape[1]*closestIndicies.shape[0],"edges.")

    G=nx.Graph()
    
    grid = np.indices(closestIndicies.shape)
    VEStack = np.stack((grid[0], closestIndicies, closestDis), axis=-1)
    VEList = VEStack.reshape(VEStack.shape[0]*VEStack.shape[1], 3).tolist()
    VETupleList = [(int(x[0]), int(x[1]), x[2]) for x in VEList]
    
    G.add_weighted_edges_from(VETupleList)
    G = nx.minimum_spanning_tree(G)
    
    return G


# This function will collect the neighbors of PStar and return a list of this points's index
# Param: PStar: the index of the point that we want to find its neighbors
#        H: the searching range for the neighbors
# Return: A: A set of points' indicies representing the neighbors
# This function will also maintain the dictionary of the distance between points and the weight 
# between points. 

def collectPointsLite(PStar: int, H:int, H_outer:int):
    
    global graph_base
    global pointsCor3D_base
    
    toExplore = [PStar]
    explored = []
    A = [PStar]
    
    while len(toExplore) > 0:
        curP = toExplore[0]
        del toExplore[0]
        explored.append(curP)
        
        for Pj in graph_base.neighbors(curP):
            # Get the distance value precomputed in the nearest neighbor process
            PjCurDist = distance.euclidean(pointsCor3D_base[Pj], pointsCor3D_base[PStar])
            if (Pj) not in A and PjCurDist < H:                     
                toExplore.append(Pj)
                A.append(Pj)
            elif (Pj) not in A and (Pj) not in explored and PjCurDist < H_outer:
                toExplore.append(Pj)    
        
    #print(counter)
    return np.asarray(pointsCor3D_base[A])

# This function is used to reconstruct the end point of the centerline after cleaning

def deleteChild(child:int):
    global removedNodeDict
    global graph_centerline
    global pointsCor3D_centerline
    
    graph_centerline.remove_node(child)
    
    for grandChild in removedNodeDict[child]:
        if(graph_centerline.has_node(grandChild)):
            deleteChild(grandChild)


# This function is used to reconstruct the end point of the centerline after cleaning

def addBackChildren(parent:int, curDepth:int):
    global removedNodeDict
    global graph_centerline
    global pointsCor3D_centerline
    
    if(parent not in removedNodeDict):
        return curDepth
    
    if(len(removedNodeDict[parent]) == 1):
        child = removedNodeDict[parent][0]
        parent_cor = pointsCor3D_centerline[parent]
        child_cor = pointsCor3D_centerline[child]
        graph_centerline.add_edge(parent, child , weight=distance.euclidean(parent_cor, child_cor))
        return addBackChildren(child, curDepth + 1)
    
    else:
        maxDepth = 0
        curChild = -1
        
        for child in removedNodeDict[parent]:
            parent_cor = pointsCor3D_centerline[parent]
            child_cor = pointsCor3D_centerline[child]
            graph_centerline.add_edge(parent, child , weight=distance.euclidean(parent_cor, child_cor))
            childDepth = addBackChildren(child, curDepth + 1)
            
            if(childDepth < maxDepth):
                deleteChild(parent, child)
            else:
                maxDepth = childDepth
                
                if(curChild != -1):
                    deleteChild(curChild)
                    
                curChild = child           
        return maxDepth

# This function is used for visulize the centerline in VPython
def pointCorToVector(pointCor):
    x = pointCor[0]
    y = pointCor[1]
    z = pointCor[2]
    return vector(x, y, z)



def getMoveVec(targetPointCor, neighborsCor):
    totalVec = neighborsCor - targetPointCor
    averageVec = np.sum(totalVec, axis=0)/len(totalVec)
    return averageVec

'''
This should return a 2D array
'''
def GenerateCenterlineCoordinates(path_to_the_png_masks, use_moving_lease_square = False, verbose = 1):
    #initialize globals

    global graph_base 
    global pointsCor3D_base 
    global removedNodeDict 
    global graph_centerline 
    global pointsCor3D_centerline 

    warnings.filterwarnings("ignore")
    filePath = path_to_the_png_masks + "/*.png"

    pointData = ReadPointFromPNG(filePath, 0, 2)
    
    sampleNum = min(7000, int(len(pointData)/10))
    
    (graph, pointsCor3D) = getMSTFromDataPoint(pointData, sampleNumber=sampleNum, maxNeighborDis=25)

    # Check if the point cloud graph is connected or not
    if(not nx.is_connected(graph)):
        raise Exception('the raw centerline points cannot form a connected MST, raise maxNeighborDis and try again!')

    # Make deepcopies of the point cloud
    pointsCor3D_base = copy.deepcopy(pointsCor3D)
    pointsCor3D_moved = copy.deepcopy(pointsCor3D)
    graph_base = copy.deepcopy(graph)

    pointsCor3D_centerline = []

    # This the new version (5/18) of the method to find the centerline points
    # The total number of iterations
    for iteration in range(3):
        H_ini = 17
        H_delta = 5
        H_cur = 0
        trial_limit = 10
        min_neighbors = 60 + iteration * 25
        max_neighbors = 850

        if(verbose):
            print("Iteration:", iteration)
            bar = progressbar.ProgressBar(max_value=len(pointsCor3D_base))

        # We will move all the points for each iteration
        for targetPoint in range(len(pointsCor3D_base)):
            H_cur = H_ini
            trial = 0
            neighborsCor = []

            while(len(neighborsCor) < min_neighbors and trial < trial_limit and not len(neighborsCor) > max_neighbors):
                neighborsCor = collectPointsLite(targetPoint, H_cur, 2*H_cur)
                H_cur += H_delta
                trial += 1

            if(trial < trial_limit):
                moveVec = getMoveVec(pointsCor3D_base[targetPoint], neighborsCor)   
                vecLen = np.linalg.norm(moveVec)

                if(vecLen > H_cur/5):
                    pointsCor3D_moved[targetPoint] = moveVec.tolist() + pointsCor3D_base[targetPoint]

            if(verbose):
                bar.update(targetPoint)

        curNeighborDis = 20
        first_flag = True
        
        while (first_flag or not nx.is_connected(graph_base)):

            if(first_flag):
                first_flag = False

            (graph_base, pointsCor3D_moved) = getMSTFromDataPoint(pointsCor3D_moved, drawMST=False, sampleNumber=len(pointsCor3D_base), maxNeighborDis=curNeighborDis)
            curNeighborDis *= 2 
            
        pointsCor3D_base = copy.deepcopy(pointsCor3D_moved)

        if(verbose):
            print("(Debug Log): iteraction count: " + str(iteration + 1))

        #Uncomment this when you run the function in jupyter notebook.
        #This function is used to display points using pptk viewer. S
        #displayPoints(pointsCor3D_base, 0.5)

    pointsCor3D_centerline = pointsCor3D_moved
    
    #Below is to trim the centerline
    graph_centerline = graph_base

    if(not nx.is_connected(graph_centerline)):
        raise Exception('the raw centerline points cannot form a connected MST, raise maxNeighborDis and try again!')

    # constantly delete the node that only has one edge, until there are only two nodes only having one edge left,
    # both of them are the endpoints of one singal path representing the colon
    toRemove = []
    removeCount = 0
    removedNodeDict = defaultdict(list)

    if (verbose):
        print("MST has", len(pointsCor3D_centerline), "nodes. Now begin to trim the graph.")

    while (True):
        toRemove = []
        for node in graph_centerline.nodes():
            if(len(graph_centerline.edges(node)) == 1):
                removedNodeDict[list(graph_centerline.edges(node))[0][1]].append(node)
                toRemove.append(node)
        if(len(toRemove) == 2):
            break
        for node in toRemove:
            graph_centerline.remove_node(node)
            removeCount += 1
            toRemove = []
            
    endpoints = toRemove

    if (verbose):
        print("Done! Trimed", removeCount, "nodes. Now MST has", len(graph_centerline.nodes), "nodes left.")
        print("Now begin reconstruct endpoints")

    # now add back the nodes that got deleted during the triming
    addBackChildren(endpoints[0], 0)
    addBackChildren(endpoints[1], 0)

    if (verbose):
        print("Done! Now MST has", len(graph_centerline.nodes), "nodes left.")

    # check if there is more than 2 endpoints
    new_endpoints = []
    for node in graph_centerline.nodes:
        if(len(graph_centerline.edges(node)) == 1):
           new_endpoints.append(node)
    if(len(new_endpoints) != 2):
        print("Fatal error: multiple endpoints detected!")

    # check if there is more than 2 path
    path = list(nx.all_simple_paths(graph_centerline, source=new_endpoints[0], target=new_endpoints[1]))
    if(len(path) != 1):
        print("Fatal error: multiple path detected!")
        
    pointsInorder = path[0]

    #Saving the computer center line coordinates            
    pointsCorInorder = []
    for point, index in zip(pointsInorder, range(len(pointsInorder))):
        pointsCorInorder.append([pointsCor3D_centerline[point], index])

    samplePointsCorInorder = np.asarray(sample(pointsCorInorder, int(len(pointsCorInorder)/2)))
    samplePointsCorInorder = sorted(samplePointsCorInorder, key=lambda x:x[1])
    samplePointsCorInorder = [x[0] for x in samplePointsCorInorder]
                            
    # displayPoints(samplePointsCorInorder, 1.3)

    save_file_name = "centerlinepoints" + time_stamp
    csv_file_name = save_file_name + ".csv"
    np.savetxt(csv_file_name, samplePointsCorInorder, delimiter=" ")
    np.save(save_file_name, samplePointsCorInorder)

    return (samplePointsCorInorder, (save_file_name + ".npy"))



