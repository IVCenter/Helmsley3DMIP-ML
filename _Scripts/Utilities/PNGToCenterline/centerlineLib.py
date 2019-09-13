import numpy as np
import pptk
import networkx as nx
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
from PIL import Image 
import progressbar

class centerlineComputor:

    def __init__(self):
        self._graph_base = None
        self._pointsCor3D_base = None
        self._removedNodeDict = None
        self._graph_centerline = None
        self._distanceTime = 0
        self._distanceDict = None
        self._currentViewer = pptk.viewer([])

    # This function will invoke a pptk viewer to render the points 
    # Param: data: an array of points coordinate. Each point's coordinate should has the format of [x, y, z]
    #       pointSize: the size of the point to be rendered on the screen
    # Return: none

    def _displayPoints(self, data, pointSize):
        v = pptk.viewer(data)
        v.set(point_size=pointSize)

    def _updatesPoints(self, data, pointSize):
        v = pptk.viewer(data)
        v.set(point_size=pointSize)


    # This function will read a series of PNG file and convert its content to point data, which is an array of points coordinate. 
    # Each point's coordinate will have the format of [x, y, z]
    # Param: filePath: the file path of the PNG files. Each file should be named as 1.png, 2.png, 3.png ... etc. All the png file should 
    #                  be ordered by the their topological order from their original dicom file
    #        orientation: 0, 1, or 2. 0 stands for coronal. 1 stands for transverse. 2 stands for sagittal.
    # Return: an array of points coordinate. Each point's coordinate has the format of [x, y, z]

    def ReadPointFromPNG(self, pngPath, orientation:int, padding:int):
        
        path_list = [im_path for im_path in glob.glob(pngPath)]
        path_list_parsed = [re.split('\\\\|\.', path) for path in path_list]
        path_list_parsed_valid = [x for x in path_list_parsed if x[-1] == 'png']
        path_list_parsed_valid = sorted(path_list_parsed_valid, key=lambda x:int(x[-2]))
        imageData = []

        for path in path_list_parsed_valid:
            s = "\\"
            path = [f for f in path if f != '']
            s = s.join(path)
            s = s[:-4] + '.png'
            imageArray = np.array(Image.open(s))
            
            for i in range(padding):
                imageData.append(imageArray)
        
        
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

    # Given the data points, return the MST of the point cloud generated from the PNG files and an array 
    # of points positions
    # Param: data: an array of points coordinate. Each point's coordinate should has the format of [x, y, z]
    #        drawMST: boolean value. Default is false. If set true, the function will also draw a MST graph at the end
    #        sampleNumber: int value. Default is 5000. This function will only sample <sampleNumber> points from the data
    # Return: a NetworkX graph representing the Minimum spanning tree of the data points

    def getMSTFromDataPoint(self, data, drawMST: bool=False, sampleNumber: int=5000,  maxNeighborDis:int=10):
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
        #self._displayPoints(sample_data, 1.3)
        
        #Create a networkX graph instance that represent MST
        print("---------------")
        print("Begin creating a MST of the sampled points cloud")
        MST = self.CreateMSTGraph(sample_data, maxNeighborDis)
        print("---------------")
        print("MST creation Done!")
        
        if(drawMST):
            nx.draw(MST, dict(enumerate(sample_data[:, :2])))
            
        return (MST, sample_data)

    # This function is used to limited the number of edges in the original graph.
    # Instead of creating a graph with full connectivity, this function will return 
    # a list of neighbor points for each point and we will only connect them in the graph
    # Param: pointsData: an array of points coordinate. Each point's coordinate has the format of [x, y, z]
    # return: a tuple(closestIndices, closesDis). ClosestIndices is a matrix of each point's neighbors. 
    #         closestDis is a matrix of the distances between each point and their neighbors

    def getNearbyPoints(self, pointsData, maxNeighborDis):
        self._distanceDict = distance.cdist(pointsData, pointsData)
        closestIndicies = np.argsort(self._distanceDict, axis=1)
        closestDis = np.sort(self._distanceDict, 1)
        threshold = maxNeighborDis # This number can be changed. The greater this number, the more edges
        return (closestIndicies[:, 1:threshold], closestDis[:, 1:threshold])

    # This function converts points' coordinate data into a minimum spanning tree. In this graph, the nodes are the points
    # from the points cloud and the edges are the connection between each point and their neighbors. The weights are each 
    # connection's distance in space
    # Param: pointsData: an array of points coordinate. Each point's coordinate has the format of [x, y, z]
    # Return: A networkX instance containing the MST

    def CreateMSTGraph(self, pointsData, maxNeighborDis):
        print("---------------")
        print("Begin calculating nearby points for each point")
        nearbyInfo = self.getNearbyPoints(pointsData, maxNeighborDis)
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

    # This function will collect the neighbors of PStar and return a list of this points's index
    # Param: PStar: the index of the point that we want to find its neighbors
    #        H: the searching range for the neighbors
    # Return: A: A set of points' indicies representing the neighbors
    # This function will also maintain the dictionary of the distance between points and the weight 
    # between points. 

    def collectPointsLite(self, PStar: int, H:int, H_outer:int):
        
        toExplore = [PStar]
        explored = []
        A = [PStar]
        addedPts = 0
        exploredNotAdded = 0
        
        while len(toExplore) > 0:
            curP = toExplore[0]
            toExplore = toExplore[1:]
            explored.append(curP)
            
            for Pj in self._graph_base.neighbors(curP):
                time7 = time.time()
                #PjCurDist = distance.euclidean(self._pointsCor3D_base[Pj], self._pointsCor3D_base[PStar])
                PjCurDist = self._distanceDict[Pj][PStar]
                time8 = time.time()
                self._distanceTime += time8 - time7
                
                if (Pj) not in A and PjCurDist < H:                     
                    toExplore.append(Pj)
                    A.append(Pj)
                    addedPts += 1
                elif (Pj) not in A and (Pj) not in explored and PjCurDist < H_outer:
                    toExplore.append(Pj)  
                    exploredNotAdded += 1  
        
        #print(addedPts, exploredNotAdded)
        return np.array(self._pointsCor3D_base[A])

    # This function is used to reconstruct the end point of the centerline after cleaning

    def deleteChild(self, child:int):
        
        self._graph_base.remove_node(child)
        
        for grandChild in self._removedNodeDict[child]:
            if(self._graph_base.has_node(grandChild)):
                self.deleteChild(grandChild)

    # This function is used to reconstruct the end point of the centerline after cleaning

    def addBackChildren(self, parent:int, curDepth:int):

        if(parent not in self._removedNodeDict):
            return curDepth
        
        if(len(self._removedNodeDict[parent]) == 1):
            child = self._removedNodeDict[parent][0]
            parent_cor = self._pointsCor3D_base[parent]
            child_cor = self._pointsCor3D_base[child]
            self._graph_base.add_edge(parent, child , weight=distance.euclidean(parent_cor, child_cor))
            return self.addBackChildren(child, curDepth + 1)
        
        else:
            maxDepth = 0
            curChild = -1
            
            for child in self._removedNodeDict[parent]:
                parent_cor = self._pointsCor3D_base[parent]
                child_cor = self._pointsCor3D_base[child]
                self._graph_base.add_edge(parent, child , weight=distance.euclidean(parent_cor, child_cor))
                childDepth = self.addBackChildren(child, curDepth + 1)
                
                if(childDepth < maxDepth):
                    self.deleteChild(parent, child)
                else:
                    maxDepth = childDepth
                    
                    if(curChild != -1):
                        self.deleteChild(curChild)
                        
                    curChild = child           
            return maxDepth

    def getMoveVec(self, targetPointCor, neighborsCor):
        totalVec = neighborsCor - targetPointCor
        averageVec = np.sum(totalVec, axis=0)/len(totalVec)
        return averageVec

    def getCenterline(self, filePath: str, numIteration:int, usePptk:bool=False):

        filePath = ".\\New_820_Lava_Cor_pre_mask\*.png"
        pointData = self.ReadPointFromPNG(filePath, 0, 1)
        (graph, pointsCor3D) = self.getMSTFromDataPoint(pointData, drawMST=False, sampleNumber=7000, maxNeighborDis=25)

        if(not nx.is_connected(graph)):
            raise Exception('the raw centerline points cannot form a connected MST, raise maxNeighborDis and try again!')

        if(usePptk):
            self._displayPoints(pointsCor3D, 0.5)
        
        self._pointsCor3D_base = copy.deepcopy(pointsCor3D)
        pointsCor3D_moved = copy.deepcopy(pointsCor3D)
        self._graph_base = copy.deepcopy(graph)

        # This the new version (5/18) of the method to find the centerline points
        # The total number of iterations
        moved_count = []

        for iteration in range(numIteration):
            print("Start centerline iteration:", iteration)
            H_ini = 17
            H_delta = 5
            H_glo = 0
            trial_limit = 10
            min_neighbors = 60 + iteration * 25
            max_neighbors = 850
            curMoved = 0
        
            # We will move all the points for each iteration

            whileLoopTime = 0
            collectTime = 0
            getMoveVecTime = 0

            with progressbar.ProgressBar(max_value=len(self._pointsCor3D_base)) as bar:
                counter = 0

                for targetPoint in range(len(self._pointsCor3D_base)):
                    H_glo = H_ini
                    trial = 0
                    neighborsCor = []
                    time1 = time.time()

                    while(len(neighborsCor) < min_neighbors and trial < trial_limit and not len(neighborsCor) > max_neighbors):
                        time3 = time.time()
                        neighborsCor = self.collectPointsLite(targetPoint, H_glo, 2*H_glo)
                        time4 = time.time()
                        collectTime += time4 - time3
                        H_glo += H_delta
                        trial += 1

                    time2 = time.time()
                    whileLoopTime += time2 - time1

                    if(trial < trial_limit):
                        time5 = time.time()
                        moveVec = self.getMoveVec(self._pointsCor3D_base[targetPoint], neighborsCor)  
                        time6 = time.time() 
                        getMoveVecTime += time6 - time5
                        #print(targetPoint, "trial:",trial,len(neighborsCor))
                        #print(moveVec)
                        vecLen = np.linalg.norm(moveVec)

                        if(vecLen > H_glo/5):
                            pointsCor3D_moved[targetPoint] = moveVec.tolist() + self._pointsCor3D_base[targetPoint]
                            #print("Moved!", vecLen, ">", H_glo/5)
                            curMoved += 1
                            
                        #else:
                            #print("Not Moved!", vecLen, "<", H_glo/5)
                    else:
                        print("Overflow!")
                    
                    bar.update(counter)
                    counter += 1
            print(whileLoopTime, collectTime, getMoveVecTime, self._distanceTime)
            curNeighborDis = 20
            first_flag = True
            recalculateCounter = 0
            
            while (first_flag or not nx.is_connected(self._graph_base)):
            
                if(first_flag):
                    first_flag = False
                    
                (self._graph_base, pointsCor3D_moved) = self.getMSTFromDataPoint(pointsCor3D_moved, drawMST=True, sampleNumber=len(self._pointsCor3D_base), maxNeighborDis=curNeighborDis)
                curNeighborDis *= 2
                recalculateCounter += 1 

                if(recalculateCounter >= 3):
                    print("Can't produce a MST graph.")
                    exit()
                
            self._pointsCor3D_base = copy.deepcopy(pointsCor3D_moved)
            self._displayPoints(pointsCor3D_moved, 0.5)
            moved_count.append(curMoved)
        print(moved_count)

        # Construct a MST only using the 750 points modified above

        if(not nx.is_connected(self._graph_base)):
            raise Exception('the raw centerline points cannot form a connected MST, raise maxNeighborDis and try again!')

        # constantly delete the node that only has one edge, until there are only two nodes only having one edge left,
        # both of them are the endpoints of one singal path representing the colon

        toRemove = []
        removeCount = 0
        self._removedNodeDict = defaultdict(list)

        print("MST has", len(self._pointsCor3D_base), "nodes. Now begin to trim the graph.")

        while (True):
            toRemove = []

            for node in self._graph_base.nodes():
                if(len(self._graph_base.edges(node)) == 1):
                    self._removedNodeDict[list(self._graph_base.edges(node))[0][1]].append(node)
                    toRemove.append(node)
            
            # Break the loop if there are only two endpoints left in the graph
            if(len(toRemove) == 2):
                break

            for node in toRemove:
                self._graph_base.remove_node(node)
                removeCount += 1
                toRemove = []
        
        endpoints = toRemove
        print("Done! Trimed", removeCount, "nodes. Now MST has", len(self._graph_base.nodes), "nodes left.")

        print("Now begin reconstruct endpoints")
        # now add back the nodes that got deleted during the triming
        self.addBackChildren(endpoints[0], 0)
        self.addBackChildren(endpoints[1], 0)

        print("Done! Now MST has", len(self._graph_base.nodes), "nodes left.")

        # Displat the points on the centerline

        to_display = []
        for node in self._graph_base.nodes:
            to_display.append(self._pointsCor3D_base[node])

        # check if there is more than 2 endpoints
        new_endpoints = []
        for node in self._graph_base.nodes:
            if(len(self._graph_base.edges(node)) == 1):
                new_endpoints.append(node)
        if(len(new_endpoints) != 2):
            print("Fatal error: multiple endpoints detected!")

        # check if there is more than 2 path
        path = list(nx.all_simple_paths(self._graph_base, source=new_endpoints[0], target=new_endpoints[1]))
        if(len(path) != 1):
            print("Fatal error: multiple path detected!")
        
        pointsCorInorder = []

        for point, index in zip(path[0], range(len(path[0]))):
            pointsCorInorder.append([self._pointsCor3D_base[point], index])

        samplePointsCorInorder = np.asarray(sample(pointsCorInorder, int(len(pointsCorInorder)/2)))
        samplePointsCorInorder = sorted(samplePointsCorInorder, key=lambda x:x[1])
        samplePointsCorInorder = [x[0] for x in samplePointsCorInorder]
                                
        self._displayPoints(samplePointsCorInorder, 1.3)
            
        return samplePointsCorInorder


            