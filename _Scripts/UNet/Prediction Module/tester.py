'''
	This class gives an example of how to use Helmesley3DMIP module
'''

from Helmesley3DMIP import MLObject

# Create an object
ob = MLObject()

# function is like GetPredictedMasksFromDicom(self, path_to_dicom = "", path_to_output = "", path_to_hdf5 = "", verbose = 0):
# if any path is equal to "", I will use default path and default hdf5 for testing purpose
results = ob.GetPredictedMasksFromDicom(verbose=1)

# after running above function, predicted masks png is saved in "path_to_output" folder.
# and the "results" above should be the numpy array for the predicted masks.