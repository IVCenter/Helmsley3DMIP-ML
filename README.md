# Automatic 3D Colon Segmentation based on UNet
A repo for 3D medical MRI/CT image visualization and organ (colon) segmentation for Crohn's Disease using Convolutional Neural Network or other deep learning alternatives.


## Quick Access to Relevant Components of this Project

* [Google Drive Folder with the Assets for Training](https://drive.google.com/drive/folders/1AunUYgQ-9ka_B1l2Z9-GuUamAn2uUq7t?usp=sharing)
* [The Paper for Submission to MICCAI 2019 Conference](https://github.com/RussellXie7/MICCAI_paper)
* [Our Collective Doc about Working with Nautilus Clusters using Kubernetes](https://docs.google.com/document/d/1wqA_Z3cJzHDX2bTvzgSnVFPjCa8qKwpf5X6XLIMTaA8/edit?usp=sharing)
* [Presentation Slides]()


## Index for the Processed Data and Our Results

### Raw Datasets

#### Dicom Files
* [2012 Dicom]() - 48 Images
* [2016 Dicom]() - 144 Images
* [2017 Dicom]() - 116 Images

#### Converted PNG format
* [2012 PNG]() - 48 Images
* [2016 PNG]() - 144 Images
* [2017 PNG]() - 116 Images

### Manual Segmentation Files

#### PNG Files
* [2012 Masks PNG]()
* [2016 Masks PNG]()
* [2017 Masks PNG]()

#### Dicom Files
* [2012 Masks Dicom]()
* [2016 Masks Dicom]()
* [2017 Masks Dicom]()

#### Combined 2x2 Grid Visualization of Segmented Images
* [2012 Combined]()
* [2016 Combined]()
* [2017 Combined]()
* [Selected 25 Images Combined]()

#### nrrd Files for 3D Slicer
* [nrrd]()


### Training Results

#### Saved Models that can be used for Predcition
* [Model #1]() - Feb 26 Model Trained with 25 Images (ReLU, 20000 Iter/ 5 Epochs, Batch=2)
* [Model #2]() - Mar 7 Model Trained with 144 Images from 2016 (ReLU, 20000 Iter/ 5 Epochs, Batch=2)

#### Testing Results Collected for Analyzing
* [2017 MRI Dataset Prediction from Model #1]()
* [2016 MRI Dataset Prediction from Model #1]()
* [2012 MRI Dataset Prediction from Model #2]()

## Built With

* [Pickle]()
* [Python 2.7](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [PyDicom]()
* [Keras with Tensorflow Kernel]()


## Developers Team

* **Wanze (Russell) Xie** - (https://github.com/russellxie7)
* **Xiaofan Lin** - ()
* **Zifeng Li** - ()


## Comments

* All the raw and processed data include the png/dicom segmented masks and nrrd files for 3D slicer (data for training): https://drive.google.com/drive/u/2/folders/130arK9MAYJD8Pq-1-6TU2y9fOTO5rJ5n
* Saved models: https://drive.google.com/drive/u/2/folders/1IM0TxECQgww1rflAW-mgwIhlyKa5Qbw3
