#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pptk
import networkx as nx
import imageio
import glob
import re
from random import sample
from scipy.spatial import distance
from time import strftime
import os
import scipy.optimize as optimize
import math
from sklearn.linear_model import LinearRegression
import time


# In[ ]:


if os.name == 'nt': # Windows
    system_win = 1
else:
    system_win = 0


# In[ ]:


# given the file path of PNGs, return the MST of the point cloud generated from the PNG files and an array 
# of points positions
def getMSTFromPNG(filePath, drawMST: bool=False, sampleNumber: int=5000):
    # Read points data from PNGs 
    print("Begin reading PNG files and convert them to point clouds")
    data = ReadPointFromPNG(filePath)
    print("---------------")
    print("Done!")
    if(sampleNumber > len(data)):
        sampleNumber = len(data)
        
    # default sample 5000 points from the whole set, otherwise it would take too long
    print("---------------")
    print("There are " + str(len(data)) + " points in total. Now sampleling " + str(sampleNumber) + " points from them")
    sample_data = np.asarray(sample(data, sampleNumber))
    print("---------------")
    print("Done!")
    #display the points 
    #displayPoints(sample_data, 1.3)
    #Create a networkX graph instance that represent MST
    print("---------------")
    print("Begin creating a MST of the sampled points cloud")
    MST = CreateMSTGraph(sample_data)
    print("---------------")
    print("MST creation Done!")
    if(drawMST):
        nx.draw(MST, dict(enumerate(sample_data[:, :2])))
    return (MST, sample_data)
    


# In[ ]:


def displayPoints(data, pointSize):
    v = pptk.viewer(data)
    v.set(point_size=pointSize)


# In[ ]:


def readPointFromTXT(filepath):
    data = np.genfromtxt(fname=filepath, skip_header=0)
    return data


# In[ ]:


def ReadPointFromPNG(filepath):
    path_list = [im_path for im_path in glob.glob(filepath)]
    if system_win:
        path_list_parsed = [re.split('\\\\|\.', path) for path in path_list]
    else:
        path_list_parsed = [re.split('/|\.', path) for path in path_list]
    path_list_parsed_valid = [x for x in path_list_parsed if x[-1] == 'png']
    path_list_parsed_valid = sorted(path_list_parsed_valid, key=lambda x:int(x[-2]))
    data_valid = []
    delta = 0.5
    thickness = len(path_list_parsed_valid) * 3 * delta
    for path in path_list_parsed_valid:
        s = ""
        if system_win:
            s = "\\"
        else:
            s = "/"
        s = s.join(path)
        s = s[:-4] + '.png'
        image = imageio.imread(s)
        for row in range(len(image)):
            for col in range(len(image)):
                if image[row][col] > 100:
                    data_valid.append([row, col, thickness])
        thickness -= 3*delta
    return data_valid


# In[ ]:


# This function is used to limited the number of edges in the original graph.
# Instead of creating a graph with full connectivity, this function will return 
# a list of neighbor points for each point and we will only connect them in the graph
def getNearbyPoints(pointsData):
    D = distance.squareform(distance.pdist(pointsData))
    closestIndicies = np.argsort(D, axis=1)
    closestDis = np.sort(D, 1)
    threshold = 10 # This number can be changed. The greater this number, the more edges
    return (closestIndicies[:, 1:threshold], closestDis[:, 1:threshold])


# In[ ]:


def CreateMSTGraph(pointsData):
    print("---------------")
    print("Begin calculating nearby points for each point")
    nearbyInfo = getNearbyPoints(pointsData)
    print("---------------")
    print("Nearby points calculation Done!")
    print("---------------")
    print("Begin construct graph")
    G=nx.Graph()
    closestIndicies = nearbyInfo[0]
    closestDis = nearbyInfo[1]
    for firstPIndex in range(len(closestIndicies)):
        for second in range(len(closestIndicies[firstPIndex])):
            secondPIndex = closestIndicies[firstPIndex][second]
            G.add_edge(firstPIndex, secondPIndex , weight = closestDis[firstPIndex][second])
    print("---------------")
    print("Graph construction Done!")
    print("---------------")
    print("Begin calculate MST")
    G = nx.minimum_spanning_tree(G)
    print("---------------")
    print("MST calculation Done!")
    return G


# In[ ]:


# Impliment the collect algorithm for 3D points in the paper
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


# In[ ]:


def collectPointsNonrec(PStar: int):
    global H_glo
    global graph
    global pointsCor3D
    global distance_dict 
    toExplore = [PStar]
    A = [PStar]
    while len(toExplore) > 0:
        curP = toExplore[0]
        del toExplore[0]
        for edge in graph.edges(curP):
            Pj = edge[1]
            if(Pj) not in A:
                if (Pj, PStar) not in distance_dict or (PStar, Pj) not in distance_dict:
                    dist_temp = distance.euclidean(pointsCor3D[Pj], pointsCor3D[PStar])
                    distance_dict[(Pj, PStar)] = dist_temp
                    distance_dict[(PStar, Pj)] = dist_temp
                if distance_dict[(Pj, PStar)] < H_glo:
                    toExplore.append(Pj)
                    A.append(Pj)
    return A


# In[ ]:


def weightFun(P1, P2):
    global distance_dict 
    if(P1 == P2):
        return 1
    return math.exp(-1 * (distance_dict[(P1, P2)]**2)/(H_glo**2))


# In[ ]:


def calculateRegressionPlane(PStar, A: list):
    global pointsCor3D
    global H_glo
    global weight_dict
    
    for point in A:
        if (PStar, point) not in weight_dict:
            weight_dict[((PStar, point))] = weightFun(PStar, point)
            weight_dict[((point, PStar))] = weightFun(PStar, point)
    
    def f(params):
        a, b, c = params 
        loss = 0
        for point in A:
            point_cor = pointsCor3D[point]
            loss += ((a*point_cor[0] + b*point_cor[1] + c - point_cor[2])**2)*weight_dict[((point, PStar))]
            #loss += ((a*point_cor[0] + b*point_cor[1] + c - point_cor[2])**2)*weightFun(PStar, point)
        return loss
    
    initial_guess = [1, 1, 1]
    result = optimize.minimize(f, initial_guess, method = 'Nelder-Mead')
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)
    return fitted_params


# In[ ]:


def projectPoints(params, A: list):
    global pointsCor3D
    a, b, c = params
    normal = np.asarray([a, b, -1])
    normal = normal / np.linalg.norm(normal)
    pointOnPlane = np.asarray([0, 0, c])
    projectionPointsCor = []
    for point in A:
        point_cor = np.asarray(pointsCor3D[point])
        pointToPlaneV = point_cor - pointOnPlane
        dist = np.dot(normal, pointToPlaneV)
        projectionPointcor = point_cor - dist*normal
        projectionPointsCor.append(list(projectionPointcor))
    return projectionPointsCor


# In[ ]:


# this function converted the 3D coordinate system of points in a plane to 2D, returns a list of new coordinates
# each of them also has x, y and z component but z is equal to 0
# this finction also will return the info of the plane, which can be used to convert a 2D coordinate to 3D again
# The format of the plane info is [u, v, origin] (u is a unit vector in 3D representing plane's x axis, y is a unit 
# vector in 3D representing plane's y axis, origin is a coordinate in 3D of plane's origin )
def convertTo2DCor(pointsCor, planeParam):
    a, b, c = planeParam
    origin = np.array([0, 0, c])
    u = np.array([0, 0, c]) - np.array([1, 1, a + b + c])
    u = u / np.linalg.norm(u)
    normal = np.array([a, b, -1])
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
    


# In[ ]:


# return the 2D coordinate of the 3D points and the information of the regression plane, which the points are located
# require gloable perameters graph and pointsCor3D
def get2DCorFrom3D(targetPoint):
    global graph
    global pointsCor3D
    '''
    global A
    A = []
    start = time.time()
    collectPoints1(targetPoint, targetPoint)
    end = time.time()
    print("collectPoints1:")
    print(start - end)
    '''
    #start = time.time()
    localPoints = collectPointsNonrec(targetPoint)
    #end = time.time()
    #print("collectPointsNonrec:")
    #print(start - end)
    #print(localPoints)
    #displayPoints(pointsCor3D[np.asarray(localPoints)], 0.5)
    #start = time.time()
    params = calculateRegressionPlane(targetPoint, localPoints)
    #print(params)
    #end = time.time()
    #print("calculateRegressionPlane:")
    #print(start - end)
    
    '''
    xyz = []
    a, b, c = params
    for x in range(0, 10):
        for y in range(0, 10):
            z = a*x + b*y + c
            xyz.append([x, y, z])
    displayPoints(xyz, 0.5)
    '''          
    #start = time.time()
    projectionPointsCor = projectPoints(params, localPoints)
    #end = time.time()
    #print("projectPoints:")
    #print(start - end)
    #start = time.time()
    points2DCor, planeInfo = convertTo2DCor(projectionPointsCor, params)
    #end = time.time()
    #print("projectPoints:")
    #print(start - end)
    return (points2DCor, planeInfo)


# In[ ]:


# This function takes a single point's 2D coordinate and transform it into 3D base on the planeInfo
def get3DCorFrom2D(pointCor, planeInfo):
    u, v, origin = planeInfo
    vectorElem1 = pointCor[0]*u
    vectorElem2 = pointCor[1]*v
    newCor = vectorElem1 + vectorElem2 + origin
    return newCor


# In[ ]:


#compue the line regression
def calculateRegressionLine(pointsCor):
    X = np.array([x[0] for x in pointsCor]).reshape(-1, 1)
    Y = np.array([x[1] for x in pointsCor]).reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    return(linear_regressor.coef_[0], linear_regressor.intercept_[0])


# In[ ]:


def rotatePointsCor(pointsCor, lineCoef):
    pointsCor = np.array(pointsCor)
    theta = math.atan(lineCoef)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([(c,-s, 1), (s, c, 1)])
    newPointsCor = []
    for point in pointsCor:
        newPointsCor.append(R.dot(point))
    return np.asarray(newPointsCor)


# In[ ]:


(graph, pointsCor3D) = getMSTFromPNG("mri_label_2016/*.png")


# In[ ]:


H_glo = 50
A = []
weight_dict = {}
distance_dict = {}
for targetPoint in range(len(pointsCor3D)):
    localPointsCor2D, planeInfo = get2DCorFrom3D(targetPoint)
    print(targetPoint, end=": ")
    #displayPoints(localPointsCor2D, 0.5)
    slope, intersept = calculateRegressionLine(localPointsCor2D)
    rotatedPointsCor = rotatePointsCor(localPointsCor2D, slope)
    #displayPoints(np.insert(rotatedPointsCor, 2, values=0, axis=1), 0.5)
    print(np.corrcoef(rotatedPointsCor[:, 0],rotatedPointsCor[:, 1])[0][1])


# In[ ]:




