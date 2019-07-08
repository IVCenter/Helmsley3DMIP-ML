# Integrated API for python

## Features included:
* automated colon mask prediction based on UNet, only need dicom files of MRI scan as input
* Denoise module to remove artifacts and false positives of the prediction to retrieve the clean predicted colon
* Centerline extraction algorithm to automatically retrieve centerline coordinates of the colon in the 3D space using the png masks of the colon, came with automatic false positive points trimming and line denoising.
* Centerline super-sampling algorithm based on Bezier curve to smooth the extracted centerline curve.
* Cross section cutting plane calculation algorithm to retrieve the U and V vectors that represent the cross section plane along the centerline on the colon in the 3D space.

## Usage
`from Helmsley3DMIP import MLObject`

## Functions
* Create instance: 
* `ob = MLObject(path_to_dicom = "MRI_Images/", path_to_hdf5 = "unet_colon_25_new_2.hdf5")`
* Predict masks: 
* `results = ob.GetPredictedMasksFromDicom(verbose = 1)`
* Get centerline: 
* `centerline_corrds = ob.GetCenterlineCoords(verbose = 1)`
* Get cross section planes: 
* `u_list, v_list, smoothed_centerline_points = ob.GetCuttingPlaneUVLists(segmentPtsDensity = 3, fwdLookLimit = 150)`

## Notes
* Download hdf5 file before running machine learning prediction for predicting masks:
* Download Link: https://drive.google.com/drive/u/1/folders/1IM0TxECQgww1rflAW-mgwIhlyKa5Qbw3

## How to play with this API
1. Run tester.py
2. Probably it will say some you didn't install some module, just pip install them
3. Run tester.py again, it should show you a progress bar in the terminal about prediction process
4. check the output folder and see the predicted images
5. check the tester.py script to see how it is using the Helmesley3DMIP module

