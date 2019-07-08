'''
	This class gives an example of how to use Helmesley3DMIP module
'''
from Helmsley3DMIP import MLObject

# Create an object and set the path to the Dicom files and the HDF5 files for the trained machine learning model
ob = MLObject(path_to_dicom = "MRI_Images/", path_to_hdf5 = "unet_colon_25_new_2.hdf5")

# function is like GetPredictedMasksFromDicom(self, path_to_dicom = "", path_to_hdf5 = "", verbose = 0):
# if any path is equal to "", I will use default path and default hdf5 for testing purpose
results = ob.GetPredictedMasksFromDicom(verbose = 0)

# after running above function, predicted masks png is saved in "path_to_output" folder.
# and the "results" above should be the numpy array for the predicted masks.
centerline_corrds = ob.GetCenterlineCoords(verbose = 0)

# Get the U list and the V list.
u_list, v_list, smoothed_centerline_points = ob.GetCuttingPlaneUVLists(segmentPtsDensity = 3, fwdLookLimit = 150)

