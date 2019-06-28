    //
//  main.cpp
//  pythonTest
//
//  Created by Zifeng Li on 6/6/19.
//  Copyright © 2019 Zifeng Li. All rights reserved.
//

#include "MLPython.hpp"
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, const char * argv[]) {

    vector<vector<float>> raw_centerline_vecs;
    vector<vector<float>> u_vecs;
    vector<vector<float>> v_vecs;
    vector<vector<float>> smooth_centerline_vecs;
    
    MLPython* myObj = new MLPython();
    string centerline_fileName = "";
    
    // Param: string path_to_dicom: the address of MRI Dicom files' folder.
    //        string path_to_hdf5: the address of the ML model. Extention is .hdf5
    //        string path_to_masks: the address of output masks' folder.
    // Function: Set the input output folders for generateMaskVolumePy()
    // This function must be called before running generateMaskVolumePy()
    // in every situation.
    // Return NULL
    myObj->ini_get_mask(".\\MRI_Images_test/", ".\\unet_colon_25_new_2.hdf5", ".\\test_output");
    
    // Param: No Param needed
    // This function will generate masks re PNG format and store them in the path_to_masks
    // folder
    // Must call ini_get_mask() before
    // Return: 1 on success, -1 on failure
    myObj->generateMaskVolumePy();
    
    // Param: string path_to_masks: the address of masks' folder
    // Function: Set the input folder for generateCenterLineCorPy()
    // This function must be called before running generateCenterLineCorPy()
    // in every situation.
    // Return NULL
    myObj->ini_get_centerline(".\\test_output");
    
    // Param: vector<vector<float>>* centerline. A referance to a vector, which
    //        will be filled with point coordinates along the centerline
    //        string* centerline_filename. A referance to a string, which will
    //        will be assigned to the name of the centerline file (.npy format)
    // Function: Return the raw centerline coordinates in std::vector and also
    //        generate a .npy file in the root directory containing the same
    //        coordinates as the returned vector
    // Must call ini_get_centerline() before
    // Return: 1 on success, -1 on failure
    myObj->generateCenterLineCorPy(&raw_centerline_vecs, &centerline_fileName);
    
    cout << "Raw centerline points count: " << raw_centerline_vecs.size() << endl;
    cout << "Centerline file name: " + centerline_fileName << endl;
    
    // Param: string npyFilePath: the address of centerline coordinate file
    // Function: Set the input centerline file for generateUVCenterFromNpy()
    // This function must be called before running generateUVCenterFromNpy()
    // in every situation.
    // Return: NULL
    myObj->ini_get_uvCenterline(centerline_fileName);
    
    // Param: 1. vector<vector<float>>* uVecs. A referance to a std::vector, which
    //        will be filled with u vectors pointing to one of the two directions
    //        defining a cross section plane
    //        2. vector<vector<float>>* vVecs. A referance to a std::vector, which
    //        will be filled with v vectors pointing to one of the two directions
    //        defining a cross section plane
    //        3. vector<vector<float>>* centerline. A referance to a std::vector,
    //        which will be filled with point coordinates along a smoothed centerline
    //        4. int segPtsDensity: Default is 3. Used for smoothing the raw centerline.
    //        It defines how mane points will be added between the raw centerline points
    //        to eventually generate a bazier curve
    //        5. int forwardLookLimit: Default is 150. Used for smoothing the uv vectors
    //        defining the cross section plane. The larger this value, the more smooth
    //        the uv vectors will be. But very large value will result in unaccurate results
    // Function: Return the uv vectors and the smoothed centerline coordinates in std::vector
    // Must call ini_get_uvCenterline() before
    // Return: 1 on success, -1 on failure
    myObj->generateUVCenterFromNpy(&u_vecs, &v_vecs, &smooth_centerline_vecs, 3, 150);
    
    cout << u_vecs.size() << " " << v_vecs.size() << " " << smooth_centerline_vecs.size() << endl;
	
    return 0;
}
