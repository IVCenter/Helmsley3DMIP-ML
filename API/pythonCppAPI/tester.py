'''
	This class gives an example of how to use Helmesley3DMIP module
'''
import pptk

from Helmesley3DMIP import MLObject

# Create an object
ob = MLObject()

# function is like GetPredictedMasksFromDicom(self, path_to_dicom = "", path_to_hdf5 = "", verbose = 0):
# if any path is equal to "", I will use default path and default hdf5 for testing purpose
results = ob.GetPredictedMasksFromDicom()

# after running above function, predicted masks png is saved in "path_to_output" folder.
# and the "results" above should be the numpy array for the predicted masks.

centerline_corrds = ob.GetCenterlineCoords()

# def displayPoints(data, pointSize):
#     v = pptk.viewer(data)
#     v.set(point_size=pointSize)

# displayPoints(centerline_corrds, 1.3)
(a,b,c) = ob.GetCuttingPlaneUVLists()

