import numpy as np
import networkx as nx
import imageio
import glob
import re
from random import sample, seed
from scipy.spatial import distance
from time import strftime
import os
import scipy.optimize as optimize
import math
from sklearn.linear_model import LinearRegression
import time
import copy
from moving_least_square import *
from tempfile import TemporaryFile
from collections import defaultdict
from math import *
import time
import datetime

seed(20)
if os.name == 'nt': # Windows
    system_win = 1
else:
    system_win = 0

H_glo = None
graph = None
pointsCor3D = None
A = None
distance_dict = None
dirty_dict = None
weight_dict = None
graph_base = None
pointsCor3D_base = None
curPlaneGuess = None
H_delta = None
min_neighbors = None
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

def getMSTFromDataPoint(data, drawMST: bool=False, sampleNumber: int=5000,  maxNeighborDis:int=10):
    # Read points data from PNGs 
    if(sampleNumber > len(data)):
        sampleNumber = len(data)
        
    # default sample 5000 points from the whole set, otherwise it would take too long
    print("---------------")
    print("There are " + str(len(data)) + " points in total. Now sampleling " + str(sampleNumber) + " points from them")
    sample_data = np.asarray(sample(list(data), sampleNumber))
    print("---------------")
    print("Done!")
    
    #display the points 
    #displayPoints(sample_data, 1.3)
    
    #Create a networkX graph instance that represent MST
    print("---------------")
    print("Begin creating a MST of the sampled points cloud")
    MST = CreateMSTGraph(sample_data, maxNeighborDis)
    print("---------------")
    print("MST creation Done!")
    
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
    print("---------------")
    print("Begin reading points data from PNG files")
    path_list = [im_path for im_path in glob.glob(filepath)]
    
    if system_win:
        path_list_parsed = [re.split('\\\\|\.', path) for path in path_list]
    else:
        path_list_parsed = [re.split('/|\.', path) for path in path_list]
    path_list_parsed_valid = [x for x in path_list_parsed if x[-1] == 'png']
    path_list_parsed_valid = sorted(path_list_parsed_valid, key=lambda x:int(x[-2]))
    
    print("There are", len(path_list_parsed_valid),"PNG files, now convert them to point coordinates in 3D")
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
        
    print("Done!")
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
    print("---------------")
    print("Begin calculating nearby points for each point")
    nearbyInfo = getNearbyPoints(pointsData, maxNeighborDis)
    closestIndicies = nearbyInfo[0]
    closestDis = nearbyInfo[1]
    print("---------------")
    print("Nearby points calculation Done! Total:", closestIndicies.shape[0], "points,",\
          closestIndicies.shape[1]*closestIndicies.shape[0],"edges.")
    print("---------------")
    print("Begin construct graph")
    G=nx.Graph()
    
    grid = np.indices(closestIndicies.shape)
    VEStack = np.stack((grid[0], closestIndicies, closestDis), axis=-1)
    VEList = VEStack.reshape(VEStack.shape[0]*VEStack.shape[1], 3).tolist()
    VETupleList = [(int(x[0]), int(x[1]), x[2]) for x in VEList]
    
    G.add_weighted_edges_from(VETupleList)
    
    '''
    for firstPIndex in range(len(closestIndicies)):
        for second in range(len(closestIndicies[firstPIndex])):
            secondPIndex = closestIndicies[firstPIndex][second]
            G.add_edge(firstPIndex, secondPIndex , weight = closestDis[firstPIndex][second])
    '''     
    print("---------------")
    print("Graph construction Done!")
    print("---------------")
    print("Begin calculate MST")
    G = nx.minimum_spanning_tree(G)
    print("---------------")
    print("MST calculation Done!")
    return G


# Impliment the collect algorithm for 3D points in the paper. This is a recursive function which may not be efficient 
# enough for the project. The non-Recursive version is right below.

def collectPoints1(P: int, PStar: int):
    global H_glo
    global graph
    global pointsCor3D
    global A

    A.append(P)
    for edge in graph.edges(P):
        Pj = edge[1]
        if(Pj) not in A and distance.euclidean(pointsCor3D[Pj], pointsCor3D[PStar]) < H_glo:
            collectPoints1(Pj, PStar)


# This function will collect the neighbors of PStar and return a list of this points's index
# Param: PStar: the index of the point that we want to find its neighbors
#        H: the searching range for the neighbors
# Return: A: A set of points' indicies representing the neighbors
# This function will also maintain the dictionary of the distance between points and the weight 
# between points. 

def collectPointsNonrec(PStar: int, H:int):
    
    global graph
    global pointsCor3D
    global distance_dict 
    global dirty_dict
    global weight_dict
    
    toExplore = [PStar]
    A = [PStar]
    distance_dict[(PStar, PStar)] = 0
    weight_dict[((PStar, PStar))] = 1
    
    while len(toExplore) > 0:
        curP = toExplore[0]
        del toExplore[0]
        for Pj in graph.neighbors(curP):
            if(Pj) not in A:
                
                # Maintain the dictionary of distance and weight between points
                if (Pj, PStar) not in distance_dict or (PStar, Pj) not in distance_dict or \
                dirty_dict[PStar] == 1 or dirty_dict[Pj] == 1:
                    dist_temp = distance.euclidean(pointsCor3D[Pj], pointsCor3D[PStar])
                    distance_dict[(Pj, PStar)] = dist_temp
                    distance_dict[(PStar, Pj)] = dist_temp
                    weight_dict[((PStar, Pj))] = weightFun(PStar, Pj)
                    weight_dict[((Pj, PStar))] = weightFun(PStar, Pj)
                    dirty_dict[Pj] = 0
                    dirty_dict[PStar] = 0
                    
                if distance_dict[(Pj, PStar)] < H:
                    toExplore.append(Pj)
                    A.append(Pj)
    return A


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
            PjCurDist = distance.euclidean(pointsCor3D_base[Pj], pointsCor3D_base[PStar])
            
            if (Pj) not in A and PjCurDist < H:                     
                toExplore.append(Pj)
                A.append(Pj)
            elif (Pj) not in A and (Pj) not in explored and PjCurDist < H_outer:
                toExplore.append(Pj)    
        
    return np.asarray(pointsCor3D_base[A])



# Calculate the weight between any tew points. This weight is used to calculate regression plane
# Params: P1: the index of the first point
#         P2: the index of the second point
# Return: the weight between the two points

def weightFun(P1, P2):
    global distance_dict 
    global dirty_dict
    global pointsCor3D
    if(P1 == P2):
        return 1
    return math.exp(-1 * (distance_dict[(P1, P2)]**2)/(H_glo**2))


# Calculate the regression plane for a specific point given its neighbors. 
# Params: PStar: the index of the point for which we want to find the gression plane.
#         A: A set of points' indicies representing PStar's neighbors
# Return: fitted_params: the regression plane's parameters, which is the A, B, C, D in Ax + By + Cz + D = 0

def calculateRegressionPlane(PStar, A: list):
    global pointsCor3D
    global H_glo
    global weight_dict
    global curPlaneGuess
    
    # The following code is used for a dynamic programming version. But currently there is no perfermance improvement 
    # using this technique. Need to explore further
    # global curScalar
    # global curALen
    '''
    weightKeyList = [(PStar, x) for x in A[curALen:]]
    wM = np.array([weight_dict[k] for k in weightKeyList])
    
    xMatrix = np.array([pointsCor3D[point][0] for point in A[curALen:]])
    yMatrix = np.array([pointsCor3D[point][1] for point in A[curALen:]])
    zMatrix = np.array([pointsCor3D[point][2] for point in A[curALen:]])
    scalarList = np.array([np.sum(xMatrix**2*wM), 2*np.sum(xMatrix*yMatrix*wM), 2*np.sum(xMatrix*wM), \
                           -2*np.sum(xMatrix*zMatrix*wM), np.sum(yMatrix**2*wM), 2*np.sum(yMatrix*wM), \
                           -2*np.sum(yMatrix*zMatrix*wM), np.sum(wM) ,-2*np.sum(zMatrix*wM), \
                           np.sum(zMatrix**2*wM)]) + curScalar
    '''
    
    weightKeyList = [(PStar, x) for x in A]
    wM = np.array([weight_dict[k] for k in weightKeyList])
    
    xMatrix = np.array([pointsCor3D[point][0] for point in A])
    yMatrix = np.array([pointsCor3D[point][1] for point in A])
    zMatrix = np.array([pointsCor3D[point][2] for point in A])
    
    def f(params):
        a, b, c, d = params 
        loss = sum(((a * xMatrix + b*yMatrix + c*zMatrix + d)**2)*wM)
        
        # Dynamic programming version
        #loss = a**2*scalarList[0] +  a*b*scalarList[1] + a*c*scalarList[2] + a*scalarList[3] + b**2*scalarList[4]\
        #+ b*c*scalarList[5] + b*scalarList[6] + c**2*scalarList[7] + c*scalarList[8] + scalarList[9]
        
        return loss
    
    result = optimize.minimize(f, curPlaneGuess, method = 'Nelder-Mead')
    
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)
        
    curPlaneGuess = fitted_params
    #curScalar = scalarList
    #curALen = len(A)
    
    return fitted_params


# This function projects a list of points to a plane specified using 'params' and return their coordinate after the
# projection in 3D
# Param: params: a plane's parameters, which is the A, B, C, D in Ax + By + Cz + D = 0
#        A: a list of points (points' indices) that need to be projected

def projectPoints(params, A: list):
    global pointsCor3D
    a, b, c, d = params
    normal = np.asarray([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    if(a != 0):
        pointOnPlane = np.asarray([-d/a, 0, 0])
    elif(b != 0):
        pointOnPlane = np.asarray([0, -d/b, 0])
    else:
        pointOnPlane = np.asarray([0, 0, -d/c])
        
    projectionPointsCor = []
    for point in A:
        point_cor = np.asarray(pointsCor3D[point])
        pointToPlaneV = point_cor - pointOnPlane
        dist = np.dot(normal, pointToPlaneV)
        projectionPointcor = point_cor - dist*normal
        projectionPointsCor.append(list(projectionPointcor))
    return projectionPointsCor


# This function converted the 3D coordinate system of points in a plane to 2D, returns a list of new coordinates
# each of them also has x, y and z component but z is equal to 0
# this finction also will return the info of the plane, which can be used to convert a 2D coordinate to 3D again
# The format of the plane info is [u, v, origin] (u is a unit vector in 3D representing plane's x axis, y is a unit 
# vector in 3D representing plane's y axis, origin is a coordinate in 3D of plane's origin )

def convertTo2DCor(pointsCor, planeParam):
    a, b, c, d = planeParam
    
    if(a != 0):
        origin = np.asarray([-d/a, 0, 0])
    elif(b != 0):
        origin = np.asarray([0, -d/b, 0])
    else:
        origin = np.asarray([0, 0, -d/c])
    
    if( a == 0 and b !=0 and c != 0):
        u = np.array([0, 0, -d/c]) - np.array([0, -d/b , 0])
        u = u / np.linalg.norm(u)
    elif( b == 0 and a !=0 and c != 0):
        u = np.array([0, 0, -d/c]) - np.array([-d/a, 0, 0])
        u = u / np.linalg.norm(u)
    elif( c == 0 and a !=0 and b != 0):
        u = np.array([0, -d/b, 0]) - np.array([-d/a, 0, 0])
        u = u / np.linalg.norm(u)
    elif(a == 0 and b == 0):
        u = np.array([1, 0, 0])
    elif(a == 0 and c == 0):
        u = np.array([0, 0, 1])
    elif(b == 0 and c == 0):
        u = np.array([0, 0, 1])
    elif(a == 0 or b == 0 or c == 0):
        print("plane parameter error! a =", a, "b =", b, "c =", c, "d =", d)
    else:
        u = np.array([0, 0, -d/c]) - np.array([1, 1, (-d - a - b)/c])
        u = u / np.linalg.norm(u)
    
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    v = np.cross(u, normal)
    v = v / np.linalg.norm(v)
    convertedPointsCor = []
    
    for pointCor in pointsCor:
        oriV = np.array(pointCor) - origin
        new_x = np.dot(oriV, u)
        new_y = np.dot(oriV, v)
        convertedPointsCor.append([new_x, new_y, 0])
        
    planeInfo = [u, v, origin]
    
    return (convertedPointsCor, planeInfo)
    

# Param: targetPoint: the index of the point that we want to find its neighbors and their coordinate in 2D 
# Return: the 2D coordinate of the 3D points and the information of the regression plane, which the points are located
# require gloable perameters graph and pointsCor3D

def get2DCorFrom3D(targetPoint):
    
    global graph
    global pointsCor3D
    global H_glo
    global H_delta
    global min_neighbors
    
    localPoints = []

    while (len(localPoints) < min_neighbors):
        localPoints = collectPointsNonrec(targetPoint, H_glo)
        if(len(localPoints) < min_neighbors):
            H_glo += H_delta

    params = calculateRegressionPlane(targetPoint, localPoints)

    projectionPointsCor = projectPoints(params, localPoints)

    points2DCor, planeInfo = convertTo2DCor(projectionPointsCor, params)

    return (points2DCor, planeInfo)


def getNeighborsCor(targetPoint):

    global pointsCor3D
    global H_glo
    global H_delta
    global min_neighbors
    
    while (len(localPoints) < min_neighbors):
        localPoints = collectPointsNonrec(targetPoint, H_glo)
        if(len(localPoints) < min_neighbors):
            H_glo += H_delta

# This function takes a single point's 2D coordinate and transform it into 3D base on the planeInfo
def get3DCorFrom2D(pointCor, planeInfo):
    u, v, origin = planeInfo
    vectorElem1 = pointCor[0]*u
    vectorElem2 = pointCor[1]*v
    newCor = vectorElem1 + vectorElem2 + origin
    
    return newCor

#compue the line regression
def calculateRegressionLine(pointsCor):
    X = np.array([x[0] for x in pointsCor]).reshape(-1, 1)
    Y = np.array([x[1] for x in pointsCor]).reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    
    return(linear_regressor.coef_[0], linear_regressor.intercept_[0])


def rotatePointsCor(pointsCor, lineCoef):
    pointsCor = np.array(pointsCor)
    theta = math.atan(lineCoef)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([(c,-s, 1), (s, c, 1)])
    newPointsCor = []
    
    for point in pointsCor:
        newPointsCor.append(R.dot(point))
    return np.asarray(newPointsCor)


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
def GenerateCenterlineCoordinates(path_to_the_png_masks, use_moving_lease_square = False, verbose = 0):
    #initialize globals
    global H_glo 
    global graph 
    global pointsCor3D 
    global A 
    global distance_dict 
    global dirty_dict 
    global weight_dict 
    global graph_base 
    global pointsCor3D_base 
    global curPlaneGuess 
    global H_delta 
    global min_neighbors 
    global removedNodeDict 
    global graph_centerline 
    global pointsCor3D_centerline 

    filePath = path_to_the_png_masks + "./*.png"

    pointData = ReadPointFromPNG(filePath, 0, 2)
    (graph, pointsCor3D) = getMSTFromDataPoint(pointData, sampleNumber=7000, maxNeighborDis=25)

    if(not nx.is_connected(graph)):
        raise Exception('the raw centerline points cannot form a connected MST, raise maxNeighborDis and try again!')

    pointsCor3D_base = copy.deepcopy(pointsCor3D)
    pointsCor3D_moved = copy.deepcopy(pointsCor3D)
    graph_base = copy.deepcopy(graph)

    pointsCor3D_centerline = []

    if (use_moving_lease_square == False):
        # This the new version (5/18) of the method to find the centerline points
        # The total number of iterations
        moved_count = []
        for iteration in range(3):
            H_ini = 17
            H_delta = 5
            H_glo = 0
            trial_limit = 10
            min_neighbors = 60 + iteration * 25
            max_neighbors = 850
            curMoved = 0
          
            # We will move all the points for each iteration
            for targetPoint in range(len(pointsCor3D_base)):
                H_glo = H_ini
                trial = 0
                neighborsCor = []
                
                while(len(neighborsCor) < min_neighbors and trial < trial_limit and not len(neighborsCor) > max_neighbors):
                    neighborsCor = collectPointsLite(targetPoint, H_glo, 2*H_glo)
                    H_glo += H_delta
                    trial += 1
                    
                if(trial < trial_limit):
                    moveVec = getMoveVec(pointsCor3D_base[targetPoint], neighborsCor)   
                    # print(targetPoint, "trial:",trial,len(neighborsCor))
                    # print(moveVec)
                    vecLen = np.linalg.norm(moveVec);

                    if(vecLen > H_glo/5):
                        pointsCor3D_moved[targetPoint] = moveVec.tolist() + pointsCor3D_base[targetPoint]
                        curMoved += 1
                
                if verbose and targetPoint % 200 == 0:
                    print ("Now runs to target point: ", targetPoint)

            curNeighborDis = 20
            first_flag = True
            
            while (first_flag or not nx.is_connected(graph_base)):
            
                if(first_flag):
                    first_flag = False
                    
                (graph_base, pointsCor3D_moved) = getMSTFromDataPoint(pointsCor3D_moved, drawMST=True, sampleNumber=len(pointsCor3D_base), maxNeighborDis=curNeighborDis)
                curNeighborDis *= 2 
                
            pointsCor3D_base = copy.deepcopy(pointsCor3D_moved)
            # displayPoints(pointsCor3D_moved, 0.5)
            moved_count.append(curMoved)

            if(verbose):
                print("(Debug Log): iteraction count: " + str(iteration + 1))

        pointsCor3D_centerline = pointsCor3D_moved;

    else:
        # the code to move points toward the centerline 
        points_centerline = []

        # The total number of iterations
        for iteration in range(2):
            points_centerline = []
            H_ini = 40
            H_delta = 5
            H_glo = 0
            trial_limit = 18
            min_neighbors = 100
            max_neighbors = 850
            correlation_threshold = 0.2 + iteration*0.05
            weight_dict = {}
            distance_dict = {}

            # Set up a dirty dictionary to record the points that has been moved
            dirty_dict = {}
            for point in range(len(pointsCor3D)):
                dirty_dict[point] = 0
            
            # We will move 750 points of all the points for each iteration
            for targetPoint in range(3000, 4000):
                
                cur_correlation = 0
                H_glo = H_ini
                trial = 0
                correlation_hist = []
                localPointsCor2D_hist = []
                planeInfo = 0
                localPointsCor2D = []
                curPlaneGuess = [1, 1, 1, 1]
                # The next two lines are used in a dynamic programming version of computing the centerline
                #curScalar = np.zeros(10)
                #curALen = 0
                
                # if the correlation of the local points is not sufficient or the number of the local points is not sufficient
                # enlarge H_glo and find neighbors again  
                while(cur_correlation < correlation_threshold and len(localPointsCor2D) < max_neighbors \
                       and trial < trial_limit):
                
                    
                    localPointsCor2D, planeInfo = get2DCorFrom3D(targetPoint)            
                    slope, intersept = calculateRegressionLine(localPointsCor2D)
                    rotatedPointsCor = rotatePointsCor(localPointsCor2D, slope)
                    cur_correlation = abs(np.corrcoef(rotatedPointsCor[:, 0],rotatedPointsCor[:, 1])[0][1])
                    # print(targetPoint, "trial",str(trial), "H =", H_glo, ":" , str(cur_correlation), \
                          # "size:", len(localPointsCor2D))
                    H_glo += H_delta
                    trial += 1
                    correlation_hist.append(cur_correlation)
                    localPointsCor2D_hist.append(copy.deepcopy(localPointsCor2D))
                    
                    
                localPointsCor2D = localPointsCor2D_hist[correlation_hist.index(max(correlation_hist))]
                centerPoint2D = np.asarray(localPointsCor2D)[:1, :2]
                newCor = Moving_Least_Square(centerPoint2D[0], np.asarray(localPointsCor2D)[:, :2])
                newCor3D = get3DCorFrom2D(newCor, planeInfo)
                # print(pointsCor3D[targetPoint], newCor3D)
                
                if(distance.euclidean(newCor3D, pointsCor3D[targetPoint]) < 80):
                    pointsCor3D[targetPoint] = list(newCor3D)
                    dirty_dict[targetPoint] = 1
                    points_centerline.append(newCor3D)

            if(verbose):
                print("(Debug Log): iteraction count: " + str(iteration + 1))

        pointsCor3D_centerline = points_centerline


    '''
    Below is to trim the centerline
    '''

    # Construct a MST only using the 750 points modified above
    #(graph_centerline, pointsCor3D_centerline) = getMSTFromDataPoint(pointsCor3D_base, drawMST=True, sampleNumber=len(pointsCor3D_base), maxNeighborDis=80)

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

    # Displat the points on the centerline

    # to_display = []
    # for node in graph_centerline.nodes:
    #     to_display.append(pointsCor3D_centerline[node])
    # displayPoints(to_display, 1.3)

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

    '''
    Saving the computer center line coordinates
    '''                    
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

    return (samplePointsCorInorder, save_file_name)


