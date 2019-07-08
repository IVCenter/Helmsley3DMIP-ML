# Automatic 3D Colon Segmentation based on UNet
A repo for 3D medical MRI/CT image visualization and organ (colon) segmentation for Crohn's Disease using Convolutional Neural Network or other deep learning alternatives.


## Quick Access to Relevant Components of this Project

* [The Google Drive Folder](https://drive.google.com/drive/folders/1AunUYgQ-9ka_B1l2Z9-GuUamAn2uUq7t?usp=sharing)
* [The Repo of the Paper for Submission to MICCAI 2019 Conference](https://github.com/RussellXie7/MICCAI_paper)
* [The Draft of the Paper](https://docs.google.com/document/d/16d-X6lfZZc0LoPJfryGSZ4GqFKdjigIY51ZcC7QM45A/edit?usp=sharing)
* [The Overleaf Docs for the Paper](https://www.overleaf.com/4492563319trcchxtphgrw)
* [Our Collective Doc about Working with Nautilus Clusters using Kubernetes](https://docs.google.com/document/d/1wqA_Z3cJzHDX2bTvzgSnVFPjCa8qKwpf5X6XLIMTaA8/edit?usp=sharing)
* [Presentation Slides](https://docs.google.com/presentation/d/16SVB5gvhoe-OGjmxUGqrrv-eT7HzAeizTpWoNYDJ4K8/edit?usp=sharing)


## Index for the Processed Data and Our Results

### Raw Datasets

#### Dicom Files
* [2012 MRI Dicom](https://drive.google.com/drive/u/1/folders/1Kq7pXDYBLuK2zVMTtTEroSwaY6LOsCH0) - 48 Images
* [2016 MRI Dicom](https://drive.google.com/drive/u/1/folders/1MQy0XIcm3zMOGPrtAOMIP2femiwhx4Ao) - 144 Images
* [2017 MRI Dicom](https://drive.google.com/drive/u/1/folders/1R3mjU86Nw_y7GGZ5j-YOubvHhVLMjSv9) - 116 Images

#### Converted PNG format
* [2012 MRI PNG](https://drive.google.com/drive/u/1/folders/1BP3lO0is7fqsVdWUr-yHg0b7d26I38Ac) - 48 Images
* [2016 MRI PNG](https://drive.google.com/drive/u/1/folders/1W5HFdBuPE9ucsEvVmdiC-E0e5p0QsZz7) - 144 Images
* [2017 MRI PNG](https://drive.google.com/drive/u/1/folders/1NhWQKBejSiJ1DWqMHefPIQiAINrDlDSH) - 116 Images

### Manual Segmentation Files

#### PNG Files
* [2012 Masks PNG]()
* [2016 Masks PNG](https://drive.google.com/drive/u/1/folders/1Tgd3OwXcL8Erp9fQAyz-D9A2-NEfqcgf)
* [2017 Masks PNG](https://drive.google.com/drive/u/1/folders/1xMdJ8vO1qOpOWRR9ravVpbH1LYP4A1bD)

#### Dicom Files
* [2012 Masks Dicom]()
* [2016 Masks Dicom](https://drive.google.com/drive/u/1/folders/1MHpbgCqto8iGzksbB66NxaYhg6rPB9oJ)
* [2017 Masks Dicom](https://drive.google.com/drive/u/1/folders/1JaKEdTkDP0C4a2AphwgbDs9qzDqskfoR)

#### Combined 2x2 Grid Visualization of Segmented Images
* [2012 Combined]()
* [2016 Combined](https://drive.google.com/drive/u/1/folders/1zibkm0_HktcdjGxqnA6h6jTfjK6qwb4_)
* [2017 Combined]()
* [Selected 25 Images Combined](https://drive.google.com/drive/u/1/folders/1zNEMqfPvCgKSDwcj-tUs9JD6uhPcTAVz)

#### nrrd Files for 3D Slicer
* [nrrd](https://drive.google.com/drive/u/1/folders/1U2Krl-tfqSd0kjTMWnLRWDKztoHZ3goP)


### Training Results

#### Saved Models that can be used for Predcition
* [Model #1](https://drive.google.com/drive/u/1/folders/1mU3aCjdGDIylqHM6U9BUqbO-p1tcOw6T) - Feb 26 Model Trained with 25 Images (ReLU, 20000 Iter/ 5 Epochs, Batch=2)
* [Model #2](https://drive.google.com/drive/u/1/folders/1xyu-3f6h6aPIVuOJzzmU_2yinqSZ6qQr) - Mar 7 Model Trained with 144 Images from 2016 (ReLU, 20000 Iter/ 5 Epochs, Batch=2)

#### Testing Results Collected for Analyzing
* [2017 MRI Dataset Prediction from Model #1](https://drive.google.com/drive/u/1/folders/1-e7w-HwYfVcwk5QmryMLBjPyRZ2ExZ_m)
* [2016 MRI Dataset Prediction from Model #1]()
* [2012 MRI Dataset Prediction from Model #2]()

## Built With

* [Pickle](https://docs.python.org/3/library/pickle.html)
* [Python 2.7](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [PyDicom](https://pydicom.github.io/pydicom/stable/index.html)
* [Keras with Tensorflow Kernel]()


## Developers Team

* **Wanze (Russell) Xie** - (https://github.com/russellxie7)
* **Xiaofan Lin** - ()
* **Zifeng Li** - ()


## Comments

* All the raw and processed data include the png/dicom segmented masks and nrrd files for 3D slicer (data for training): https://drive.google.com/drive/u/2/folders/130arK9MAYJD8Pq-1-6TU2y9fOTO5rJ5n
* Saved models: https://drive.google.com/drive/u/2/folders/1IM0TxECQgww1rflAW-mgwIhlyKa5Qbw3
