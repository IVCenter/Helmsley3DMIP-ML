from CurveRegression import *


# parse commandline param
folder_name =  

# some initialization
scene = canvas()
seed(42)
if os.name == 'nt': # Windows
    system_win = 1
else:
    system_win = 0

# read file and load points
filePath = folder_name + "/*.png"
pointData = ReadPointFromPNG(filePath, 0, 4)
(graph, pointsCor3D) = getMSTFromDataPoint(pointData, drawMST=False, sampleNumber=5000)

# show the sampled points
displayPoints(pointsCor3D, 1.3)

