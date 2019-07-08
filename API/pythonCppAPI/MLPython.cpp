//
//  MLPython.cpp
//  pythonTest
//
//  Created by Zifeng Li on 6/10/19.
//  Copyright © 2019 Zifeng Li. All rights reserved.
//
#include "MLPython.hpp"

using namespace std;

MLPython:: MLPython(){
    
	// The following code is reserved for mac only
	/*
	// Address of the python default modules
    string pythonDefaultPath = "/Users/zifengli/anaconda3/lib/python3.6";
    // Address of the python dynamic loaded modules, such as math and sys
    string pythonDynlibPath = ":/Users/zifengli/anaconda3/lib/python3.6/lib-dynload";
    // Address of this program's root directory
    string currentFolder = ":/Users/zifengli/OneDrive/CSE199/vtk_test/pythonTest/pythonTest";

    
    string appendedPaths = (string) pythonDefaultPath + pythonDynlibPath + pythonSitePackage + currentFolder;
    wchar_t* convertedPaths = Py_DecodeLocale(appendedPaths.c_str(), NULL);
    
    // Add all the paths to the environment
    Py_SetPath(convertedPaths);
	*/

	/*Modify this line to be the address of the root folder*/
	string currentFolder = ";C:\\Users\\Zifeng Li\\Desktop\\Helmsley3DMIP\\pythonCppAPI";

	/*Modify this line to be site-packages' address*/
	// Address of the pip installed modules, such as numpy and keras
	string pythonSitePackage = ";C:\\Users\\Zifeng Li\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages";

	wchar_t* pathPy = Py_GetPath();
	char* path = Py_EncodeLocale(pathPy, NULL);

	string appendedPaths = (string)path + currentFolder + pythonSitePackage;
	wchar_t* convertedPaths = Py_DecodeLocale(appendedPaths.c_str(), NULL);

	// Add all the paths to the environment
	Py_SetPath(convertedPaths);
	cout << convertedPaths << endl;

    Py_Initialize();
    
    PyObject* scriptModule = checkModule((char*) "Helmesley3DMIP");
    PyObject* constructorFunc = getFunctionFromModule(scriptModule, (char*) "MLObject");
    MLObjectPy = PyObject_CallObject(constructorFunc, NULL);
    
    if(!MLObjectPy){
        throw std::invalid_argument("Fatal Error! Cannot Create MLObject in Python!");
    }
    
    centerlineCorPy = NULL;
    mode = -1;
    
}


MLPython:: ~MLPython(){
    
    Py_DecRef(centerlineCorPy);
    Py_DecRef(MLObjectPy);
    
    Py_Finalize();
}


long MLPython::getLongFromPySize(Py_ssize_t input){
    
    PyObject* tempPy = PyLong_FromSsize_t(input);
    long output = PyLong_AsLong(tempPy);
    return output;
}


vector<float>  MLPython::getVec3(PyObject* input){
    Py_ssize_t vec3LengthPy = PySequence_Length(input);
    int vec3Length = int(getLongFromPySize(vec3LengthPy));
    
    assert(vec3Length == 3);
    
    vector<float> vec3;
    
    for(int i = 0; i < 3; i++){
        PyObject* listEle = PySequence_GetItem(input, i);
        float eleValue = PyFloat_AsDouble(listEle);
        vec3.push_back(eleValue);
    }
    return vec3;
}


PyObject*  MLPython::checkModule(char* moduleName){
    
    PyObject* myModule = PyImport_ImportModule((char*) moduleName);
    
    if(!myModule){
        
        try{
            int succeed = PyRun_SimpleString(("import " + (string) moduleName).c_str());
            
            if(succeed != 0){
                throw std::invalid_argument( "bad Function" );
            }
        }
        catch (exception &e){
            cout << "Error occured when importing " + (string) moduleName << endl;
            return NULL;
        }
    }
    
    return myModule;
}


PyObject*  MLPython::getFunctionFromModule(PyObject* module, char* funcName){
    
    PyObject* myFuncNamePy = PyUnicode_FromString(funcName);
    
    PyObject* myFunction = PyObject_GetAttr(module, myFuncNamePy);
    
    if(!myFunction){
        cout << "Error! " + (string)funcName + " Not Found in the module!" << endl;
        return NULL;
    }
    
    return myFunction;
}


int MLPython::generateUVCenterFromNpy(vector<vector<float>>* u_vecs, vector<vector<float>>* v_vecs,\
                                      vector<vector<float>>* centerline_vecs,int segPtsDensity, int forwardLookLimit){
    
    PyObject* myFuncNamePy = PyUnicode_FromString((char*) "GetCuttingPlaneUVLists");
    PyObject* myFunction = PyObject_GetAttr(MLObjectPy, myFuncNamePy);
    
    if(!myFunction){
        cout << "Error! Failed to load python function GetCuttingPlaneUVLists" << endl;
        return -1;
    }
    
    // Setting up the function arguments
    cout << "Using centerline file: " << npyPath << endl;
    
    PyObject* npyFilePathPy = PyUnicode_FromString(npyPath.c_str());
    PyObject* args = PyTuple_Pack(3, npyFilePathPy, PyLong_FromLong((long) segPtsDensity), \
                                  PyLong_FromLong((long) forwardLookLimit));
    
    // Actually call the python function with the arguments
    PyObject* myResult = PyObject_CallObject(myFunction, args);
    
    //  Unpack the return results and record them seperately
    PyObject* u_list = PySequence_GetItem(myResult, 0);
    PyObject* v_list = PySequence_GetItem(myResult, 1);
    PyObject* center_list = PySequence_GetItem(myResult, 2);
    
    // Get the length of the u_list, v_list and centers
    Py_ssize_t u_list_length_py = PySequence_Length(u_list);
    int u_list_length = int(getLongFromPySize(u_list_length_py));
    
    Py_ssize_t v_list_length_py = PySequence_Length(v_list);
    int v_list_length = int(getLongFromPySize(v_list_length_py));
    
    Py_ssize_t center_list_length_py = PySequence_Length(center_list);
    int center_list_length = int(getLongFromPySize(center_list_length_py));
    
    if(u_list_length == 0){
        cout << "Error! The length of the u, v list and center lisr are incorrect" << endl;
        cout << "Length of u_list is 0" << endl;
        return -1;
    }
    
    cout << u_list_length << endl;
    
    assert(u_list_length == v_list_length);
    assert(u_list_length == center_list_length);
    
    for(int i = 0; i < u_list_length; i++){
        PyObject* uListEle = PySequence_GetItem(u_list, i);
        PyObject* vListEle = PySequence_GetItem(v_list, i);
        PyObject* centerListEle = PySequence_GetItem(center_list, i);
        
        vector<float> u_vec = getVec3(uListEle);
        vector<float> v_vec = getVec3(vListEle);
        vector<float> centerline_vec = getVec3(centerListEle);
        
        u_vecs->push_back(u_vec);
        v_vecs->push_back(v_vec);
        centerline_vecs->push_back(centerline_vec);
    }
    
    return 1;
}


int MLPython::generateMaskVolumePy(int verbose){
    
    if(mode != 0){
        cout << "Must call ini_get_mask(string path_to_dicom, string path_to_hdf5, string path_to_masks)\
        first" << endl;
        return -1;
    }
    
    // Load python function
    PyObject* myFuncNamePy = PyUnicode_FromString((char*) "GetPredictedMasksFromDicom");
    PyObject* myFunction = PyObject_GetAttr(MLObjectPy, myFuncNamePy);
    
    if(!myFunction){
        cout << "Error! Failed to load python function GetPredictedMasksFromDicom" << endl;
        return -1;
    }
    
    // Packing function arguments
    PyObject* path_to_dicomPy = PyUnicode_FromString( dicomPath.c_str() );
    PyObject* path_to_hdf5Py = PyUnicode_FromString( hdf5Path.c_str() );
    PyObject* path_to_outputPy = PyUnicode_FromString( masksPath.c_str()) ;
    PyObject* verbosePy = PyLong_FromLong((long) verbose);
    PyObject* args = PyTuple_Pack(4, path_to_dicomPy, path_to_hdf5Py, path_to_outputPy, verbosePy);
    
    // Getting the result of the function
    PyObject* succeedPy = PyObject_CallObject(myFunction, args);
    long succeed = PyLong_AsLong(succeedPy);

    if(succeed != 1){
        cout << "Errors found in Python function. Cannot generate mask volume!" << endl;
        return -1;
    }
    
    return 1;
}


int MLPython::generateCenterLineCorPy(vector<vector<float>>* raw_centerline_vec, \
                                      string* centerline_filename, int verbose){
    
    if(mode != 1){
        cout << "Must call ini_get_centerline(string path_to_masks) first" << endl;
        return -1;
    }
    
    // Load Python function
    PyObject* myFuncNamePy = PyUnicode_FromString((char*) "GetCenterlineCoords");
    PyObject* myFunction = PyObject_GetAttr(MLObjectPy, myFuncNamePy);
    
    if(!myFunction){
        cout << "Error! Failed to load python function GetCenterlineCoords" << endl;
        return -1;
    }
    
    // Pack function arguments
    PyObject* path_to_png_masksPy = PyUnicode_FromString( masksPath.c_str() );
    PyObject* verbosePy = PyLong_FromLong((long) verbose);
    PyObject* args = PyTuple_Pack(2, path_to_png_masksPy, verbosePy);
    
    // Getting the result of the function
    PyObject* myResult = PyObject_CallObject(myFunction, args);
    
    //  Unpack the return results and record them seperately
    PyObject* raw_centerline_listPy = PySequence_GetItem(myResult, 0);
    PyObject* centerlin_fileNamePy = PySequence_GetItem(myResult, 1);
    
    if(!raw_centerline_listPy){
        cout << "Error found in Python function. Cannot generate centerline \
        coordinates!" << endl;
        return -1;
    }
    
    // Get the length of the raw centers
    Py_ssize_t raw_centerline_list_length_py = PySequence_Length(raw_centerline_listPy);
    int raw_centerline_list_length = int(getLongFromPySize(raw_centerline_list_length_py));
    
    if(raw_centerline_list_length == 0){
        cout << "Error! The length of the raw centerline list is 0" << endl;
        return -1;
    }
    
    for(int i = 0; i < raw_centerline_list_length; i++){
        PyObject* listEle = PySequence_GetItem(raw_centerline_listPy, i);
        vector<float> ptrCor = getVec3(listEle);
        raw_centerline_vec->push_back(ptrCor);
    }
    
    wchar_t* tempCenterlineName = PyUnicode_AsUnicode(centerlin_fileNamePy);
    *centerline_filename = Py_EncodeLocale(tempCenterlineName, NULL);
    
    return 1;
}

void MLPython::ini_get_mask(string path_to_dicom, string path_to_hdf5, string path_to_masks){
    
    dicomPath = path_to_dicom;
    hdf5Path = path_to_hdf5;
    masksPath = path_to_masks;
    mode = 0;
}

void MLPython::ini_get_centerline(std::string path_to_masks) {
    
    masksPath = path_to_masks;
    mode = 1;
}

void MLPython::ini_get_uvCenterline(std::string path_to_npy) {
    npyPath = path_to_npy;
    mode = 2;
}
