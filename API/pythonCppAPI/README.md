# Helmsley3DMIP Machine Learning C++ API
This API provides a C++ based interface to run the machine learning model, which is written in Python
This project can run in both Windows and Unix operating systems. 
This project used the [Python 3.6 C API](https://docs.python.org/3.6/c-api/)
## Getting Started
1. Download and install [Python 3.6.8 x86-64](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe).    **Please Select the "Add Python 3.6 to Path" on the first page**
2. Open the windows command line tool and enter ```where python``` to check whether the python 3.6.8 is correctly installed and get the path to Python's source code folder. 
Usually it will be in the format of ```C:\Users\[User Name]\AppData\Local\Programs\Python\Python36\```
3. Navigate to Python's source code folder and find the file ```python36.lib``` under the ```libs``` folder. Make a copy of this file and rename it to be ```python36_d.lib```. This is because the Python C API requires a debug version of this library, which is not included in the public build. Fortunately, by just making a copy of the public python36 library, the code in this program can still work correctly.
4. Open the Windows command line and navigate to the root folder of this project. Then enter the following pip install
```
pip install -r requirements.txt
```
This command will automatically install all the required extension module needed for this project. 

## Building The Project
This section contains the build settings for this project. We will use Microsoft Visual Studio as an example. 
This repository also include a ```main.cpp```. Please look through this file. It contains all the examples you needed for using this API
### Using Microsoft Visual Studio 
1. Create a new project by importing from the existing code.  
2. **Change the program to use x64 platform.**
3. Open the project properties. Add the ```include``` folder path of the Python3.6 to the ```Include Directories``` under the ```VC++ Directories``` tag. Normally it will be in the format of ```C:\Users\[User Name]\AppData\Local\Programs\Python\Python36\include\```.
Be sure that all the configuration is under the x64 platform.
4. Open the project property settings. Add the ```libs``` folder path of the Python3.6 to the ```Library Directories``` under the ```VC++ Directories``` tag. Normally it will be in the format of ```C:\Users\[User Name]\AppData\Local\Programs\Python\Python36\libs\```
Be sure that all the configuration is under the x64 platform.
4. Build the project. There should be no errors
## After Successfully Building
1. Open the ```MLPython.cpp``` and locate the constructor function, which is set to be the first function in this file. 
2. Locate the string variable ```string currentFolder``` and ```string pythonSitePackage``` and costimize them for your system. 
2.1 Modify the ```currentFolder``` to be the absolute path of this project's root directory.
2.2 Modify the ```pythonSitePackage``` to be the absolute path of the ```site-packages``` path of Python 3.6. Usually it will be in the format of ```C:\Users\[User Name]\AppData\Local\Programs\Python\Python36\Lib\site-packages```.
**Attention! You must put a ```;``` at the very beginning of the two strings.**

## Documentaion for the functions
All the description and usage for the functions can be found in the ```main.cpp```. Please check of the file to get more information.
### Dowload the hdf5 machine learning model for the ```generateMaskVolumePy()``` function
Because the machine learning model is too large for github. Click [here](https://drive.google.com/open?id=1YC9drMur7anmQYoI7ppio-Ap_c9JqsD4) to download the model. This model is trained using 20 pictures of the 2017 dataset and can be used on the whole dataset of the 2017 MRIs.

