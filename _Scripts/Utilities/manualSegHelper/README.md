
## How to set up the environment on your local machine to use this tool.
With these settings, you can harvest most of the GPU power of your machine.

1. Download and install [Anaconda navigator Python 3.7 version for Windows](https://www.anaconda.com/distribution/#download-section).
(Using all default setting during the installation will be fine)
2. Install [git for Windows](https://git-scm.com/download/win) if you do not have git.
(Using all default setting during the installation will be fine)
3. Update your NVIDIA graphic driver to the newest version. Recommend using [Gefource Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/).
4. Open Anaconda navigator. Select ```Home``` tag, and find the Jupyter notebook block on the right block and click ```install```.
5. Select the ```Environment``` tab on the left. You will see a list of virtual environments in your machine. You should see ```base(root)``` listed.
6. Select ```Create``` at the bottom to create a new virtual environment. 
   Name it whatever you want. Choose ```Python 3.6``` for packages. Then click ```Create```. **Attention! You have to choose Python 3.6 instead of 3.7 for it to be compatible with Tensorflow.** 
7. You will see the new environment listed in the anaconda navigator. Click the triangle button right next to the new environment and select ```Open with terminal```. This will open a terminal that is within this virtual python environment.  
8. Inside the terminal, enter the following commands to install packages. This could take a while.
   ```
    pip install tensorflow-gpu
    pip install scipy
    pip install matplotlib
    pip install scikit-image
    pip install pydicom
    
   ```
9. In a browser, use this link to download [cuda toolkit 10.0](https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10).

10. Install Cuda toolkit 10.0. Using all default installation option will be fine. This could take a while.

11. Download [NVDIA cuDNN for CUDA 10.0 package for Windows](https://developer.nvidia.com/cudnn). 
You have to register an account to download cuDNN. Follow the instructions on the webpages will be fine. **Attention! You have to choose cuDNN v7.6.2 for CUDA 10.0 for Windows in your last selection so that it will be compatible with the toolkit you download in the previous step.**
12. In the search bar next to the Windows begin button, search for ```environment variable``` and you should see a 
```Edit the system environment settings``` function. Open it and look for ```Environment Variables...``` in the ```advance``` tag. 
Click it and an environment variables windows will pop up. Look for ```CUDA_PATH``` and ```CUDA_PATH_V10_0``` variables in the system variables block at the lower part. Make sure that they all point to the same CUDA v10.0 path. Usually, it will be in the path 
```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0```

13. Double click the ```CUDA_PATH``` variable and in the pop-up window, click ```browse file```. This will open a window explorer. 
Enter the ```v10.0``` folder. Then, open the cuDNN package you downloaded, enter the ```cuda``` folder. Copy everything inside the 
```cuda``` folder to the inside of the ```v10.0``` folder.

14. Create a folder under you home folder (Usually it is ```C:\Users\<Your user name>\```, so a folder on the Desktop will be fine). **Attention! It has to be under your home directory so that a Jupyter notebook server can access it. Make sure the name of the folder does not contain space and special characters.**

15. Ender the folder and right-click to bring up the folder menu. If you have installed git, there will be an item ```Git bash here```.
Click that item and a console will pop up. Enter the following command:

    ```git clone https://github.com/CalVR/Helmsley3DMIP.git```

16. Open Anaconda Navigator, go to ```Environment``` tag and select the virtual environment you created. Click the triangle and click
```open with Jupyter Notebook```. This will bring up a browser with Jupiter notebook and the initial location will be your home
directory. Navigate to our project folder.

