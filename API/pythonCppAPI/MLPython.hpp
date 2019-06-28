//
//  MLPython.hpp
//  pythonTest
//
//  Created by Zifeng Li on 6/10/19.
//  Copyright © 2019 Zifeng Li. All rights reserved.
//

#ifndef MLPython_hpp
#define MLPython_hpp

#include <Python.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

class MLPython{
    
public:
    
    void ini_get_mask(string path_to_dicom, string path_to_hdf5, string path_to_masks);
    
    void ini_get_centerline(string path_to_masks);
    
    void ini_get_uvCenterline(string npyFilePath);
    
    int generateMaskVolumePy( int verbose = 1 );
    
    int generateCenterLineCorPy(vector<vector<float>>* raw_centerline_vec, string* centerline_filename, \
                                int verbose = 1 );
    
    int generateUVCenterFromNpy(vector<vector<float>>* u_vecs, vector<vector<float>>* v_vecs, \
                                vector<vector<float>>* center_vecs, int segPtsDensity = 3, \
                                int forwardLookLimit = 150);
    
    MLPython();
    ~MLPython();
    
private:
    
    // Mode = -1. Not Initialized
    // Mode = 0. Start from the beginning
    // Mode = 1. Already got the masks in folder
    // Mode = 2. Already got the centerline's npy file
    int mode;
    
    string dicomPath, hdf5Path, masksPath, npyPath;
    
    void clearAllState();
    
    PyObject* MLObjectPy;
    PyObject* centerlineCorPy;

    PyObject* getFunctionFromModule(PyObject* module, char* funcName);
    PyObject* checkModule(char* moduleName);
    vector<float> getVec3(PyObject* input);
    long getLongFromPySize(Py_ssize_t input);

    
};

#endif /* MLPython_hpp */
