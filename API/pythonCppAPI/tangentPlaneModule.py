import numpy as np
from scipy.spatial import distance
from math import *

def convertToPureList(u_list, v_list, center_list):
    u_list_converted = []
    v_list_converted = []
    center_list_converted = []
    
    for u, v, center in zip(u_list, v_list, center_list):
        u_list_converted.append(u.tolist())
        v_list_converted.append(v.tolist())
        center_list_converted.append(center.tolist())

    return u_list_converted, v_list_converted, center_list_converted

def SmoothSuperSampling(ordered_points, segment_points_density: int=3):
    smooth_curve_points = []
    
    num_points = len(ordered_points)
    # select the points that are will be gone through
    num_actual_points = (num_points - num_points % 2) / 2
    offset = num_points % 2
    num_actual_points = int(num_actual_points - 2 + offset)
    print ("num_actual_points:", num_actual_points)
    
    for i in range(num_actual_points):
        
        p0 = ordered_points[2 * i]
        p1 = ordered_points[2 * i + 1]
        p2 = ordered_points[2 * i + 2]
        
        if (2 * i - 1) >= 0:
            p0 = (ordered_points[2 * i - 1] + p1) / 2
        
        if (2 * i + 3) < num_points:
            p2 = (p1 + ordered_points[2 * i + 3]) / 2
        
        segment_points_count = int(distance.euclidean(p0, p2) * segment_points_density)
        smooth_curve_points += BezierCurveCreator(p0, p1, p2, segment_points_count)
    
    return smooth_curve_points

def BezierCurveCreator(p0, p1, p2, pointCount):
    pointList = []
    
    for i in range(pointCount):
        t = (i + 1) / float(pointCount)
        
        c0 = (1 - t) * (1 - t)
        c1 = 2 * (1 - t) * t
        c2 = t * t
        
        point = c0 * p0 + c1 * p1 + c2 * p2
        
        pointList.append(point)
    
    return pointList


def CollectPlanePointPair(centerline_points, forwardLookLimit):
    plane_point_pairs = []
    
    forwardLookLimit = 200
    
    i = 0
    for p in centerline_points:
        AssertSameVector(p, centerline_points[i])
        
        tangent = (0,0,0)
        tangentList = []
        forwardStep = 0
        
        if(i + forwardLookLimit < len(centerline_points)):
            forwardStep = forwardLookLimit
        else:
            forwardStep = len(centerline_points) - i - 1
        
        for j in range(forwardStep):
            tangentList.append(VectorBetweenTwoPoints(p, centerline_points[i + j + 1]))
        
        if(forwardStep == 0):
            break
        
        avgTangent = tangentList[0]
        
        for tangent in tangentList[1:]:
            avgTangent += tangent
        
        avgTangent /= len(tangentList)
        
        n = normalize(avgTangent)
        
        plane_param = GetPlaneParam(p,n)
        plane_point_pairs.append((plane_param, p))
        
        i += 1
    
    return plane_point_pairs

'''
Helper Method for CollectPlanePointPair()
'''
def AssertSameVector(v1, v2):
    assert len(v1) == len(v2)
    for i in range (len(v1)):
        assert v1[i] == v2[i]


def VectorBetweenTwoPoints(p1, p2):
    return np.array(p2 - p1)

def GetPlaneParam(p, n):
    a = n[0]
    b = n[1]
    c = n[2]
    d = - (n[0]*p[0] + n[1]*p[1] + n[2]*p[2])
    return (a,b,c,d)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

'''
Helper Method for computeYVArray()
    
'''
def ProjectVectorOnToPlane(original_v, plane_normal):
    original_v = np.array(original_v)
    plane_normal = np.array(plane_normal)
    v_project_to_normal = np.dot(original_v, plane_normal)
    v_project_to_normal = v_project_to_normal / (np.dot(plane_normal,plane_normal))
    v_project_to_normal = np.dot(v_project_to_normal, plane_normal)
    v_project_to_plane = original_v - v_project_to_normal
    return normalize(v_project_to_plane)

# This function is used to pre-compute u and v vectors for the cross section plane

def computeUVarray(plane_point_pairs):
    prev_u = [0, 0, 0]
    a, b, c, d = plane_point_pairs[0][0]
    center = plane_point_pairs[0][1]
    u = []
    v = []
    normal = np.array([a, b, c])
    normal = normalize(normal)
    
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
    
    v = np.cross(u, normal)
    v = normalize(v)
    prev_u = u
    u_list = [u]
    v_list = [v]

    for i in range(1, len(plane_point_pairs)):
        a, b, c, d = plane_point_pairs[i][0]
        centerPoint = plane_point_pairs[i][1]
        
        if(a * centerPoint[0] + b * centerPoint[1] + c * centerPoint[2] + d != 0):
            print("error: center point is not on the plane!")
            return
        
        u = []
        v = []
        normal = np.array([a, b, c])
        normal = normalize(normal)
        u = ProjectVectorOnToPlane(prev_u, normal)
        v = np.cross(u, normal)
        v = normalize(v)
        prev_u = u
        u_list.append(u)
        v_list.append(v)

    return u_list, v_list


def cFuncGetCenterUVFromPtsNpy(npyPath, segmentPtsDensity = 3, fwdLookLimit = 150):
    
    centerline_array = np.load(npyPath)
    centerline_points = SmoothSuperSampling(centerline_array, segmentPtsDensity)
    plane_point_pairs = CollectPlanePointPair(centerline_points, fwdLookLimit)
    u_list, v_list = computeUVarray(plane_point_pairs)
    u_list, v_list, centerline_points = convertToPureList(u_list, v_list, centerline_points)
    
    return (u_list, v_list, centerline_points)


