# Detect defective medicine packaging using OpenCV and TensorFlow

This repository contains 5 Jupyter Notebook files (.ipynb) that I've made and modified to implement object detection for Computer vision systems to detect defects and anomalies in medicine packaging, such as broken seals or damaged blister packs using TensorFlow.

## Table of Contents
- [Abstract](#abstract)
- [Getting Started](#getting-started)
- [Setting up the Conda Environment](#setting-up-the-conda-environment)
- [0.0 **Testing GPU.ipynb**](#00-testing-gpuipynb)
- [1.0 **Image collection.ipynb**](#10-image-collectionipynb)
- [2.0 **Split images to train-test.ipynb**](#20-split-images-to-train-testipynb)
- [2.1 **Split images (alternative).ipynb**](#21-split-images-alternativeipynb)
- [3.0 **Training and Detection.ipynb**](#30-training-and-detectionipynb)
- [Acknowledgments and License Information](#acknowledgments-and-license-information)

## <a id="0.0"></a>Abstract
Ensuring the quality of medicine packaging is essential for maintaining the safety and efficacy of pharmaceutical products. This project presents a common approach for detecting defective medicine packaging by combining the power of OpenCV and TensorFlow, two popular open-source computer vision and machine learning libraries. This method involves capturing, and preprocessing the images of medicine packaging using OpenCV to enhance the relevant features and reduce noise. As well as train a deep learning model using TensorFlow to classify the images as either defective or non-defective. The results demonstrate that this approach achieves high accuracy in detecting various types of packaging defects, such as damaged seals, or missing tablets. It is important to note that the presented work is meant for testing purposes, and similar technologies are already being utilized in the industry to address packaging defects. However, this project is a good starting point for anyone who wants to learn how to implement object detection using OpenCV and TensorFlow. 
## <a id="0.0"></a>Getting Started
To get started, clone this repository and setup the environment. Follow the instructions in each notebook to execute the code.

```bash
## Clone the repository
git clone https://github.com/Code-nano/Defective-Medicine-Packaging-Detector.git
```

## <a id="0.0"></a>Setting up the Conda Environment

I have provided an environment.yaml file to help set up a Conda environment with all the necessary dependencies. 

To create and activate the environment, follow these steps:

**1.** Install Anaconda or Miniconda if you haven't already

**2.** Create the Conda environment using the environment.yaml file:

- if you want to change the environment name, change the name in the environment.yaml file before running the following command:

    ```bash
    # Creates the Conda environment
     conda env create -f environment.yaml
    ```
**3.** Activate the new environment
- the default enviroment name is ds-c-tfod-01

    ```bash
    # activates the environment
    conda activate ds-c-tfod-01
    ```

## Notebooks
### <a id="0.0"></a>[0.0 Testing GPU.ipynb](https://github.com/Code-nano/Tensortflow_object_detection-01/blob/b31adc89934316b3a18d8f8ec942389d5d54dc02/0.0%20Testing%20GPU.ipynb)
This notebook aims to display the current versions of various Python libraries and check for GPU availability. Specifically, it reports the versions of TensorFlow, Keras, Python, Pandas, and Scikit-Learn, and verifies if a GPU device is available for TensorFlow.

### <a id="1.0"></a>[1.0 Image collection.ipynb](https://github.com/Code-nano/Tensortflow_object_detection-01/blob/b31adc89934316b3a18d8f8ec942389d5d54dc02/1.0%20Image%20collection.ipynb)
This notebook sets up a real-time image capture and processing pipeline using OpenCV. It captures images from a webcam, crops and resizes them, and saves the resulting images to disk with unique names. The pipeline also provides a countdown timer before each image is captured, which allows users to prepare for the photo. This notebook also installs and compiles the LabelImg tool, which is a graphical image annotation tool used for labeling object bounding boxes in images. It helps to create the required training data for object detection models.

#### Features
- Set up real-time image capture and processing pipeline using OpenCV
- Capture images from a webcam
- Crop and resize captured images
- Save resulting images to disk with unique names
- Provide countdown timer before each image capture
- Install and compile LabelImg tool
- Use LabelImg for labeling object bounding boxes in images
- Create training data for object detection models

### <a id="2.0"></a>[2.0 Split images to train-test.ipynb](https://github.com/Code-nano/Tensortflow_object_detection-01/blob/b31adc89934316b3a18d8f8ec942389d5d54dc02/2.0%20Split%20images%20to%20train-test.ipynb)
This notebook splits a dataset of labeled images into training and testing sets, following a specified split ratio (default is 80% for training and 20% for testing). It copies the images and their corresponding XML annotation files to separate train and test folders. The XML files are used to generate a CSV file containing the image filename and its corresponding bounding box coordinates. This CSV file is used to generate a TFRecord file, which is the required format for training object detection models.

#### Features
- Split labeled images dataset into training and testing sets
- Follow specified split ratio (default: 80% training, 20% testing)
- Copy images and corresponding XML annotation files to separate train and test folders
- Use XML files to generate CSV file with image filename and bounding box coordinates
- Create TFRecord file from CSV file
- Prepare dataset in required format for training object detection models

### <a id="2.1"></a>[2.1 split images (alternative).ipynb](https://github.com/Code-nano/Tensortflow_object_detection-01/blob/b31adc89934316b3a18d8f8ec942389d5d54dc02/2.1%20split%20images%20(alternative).ipynb)
This is the improved version of the 2.0 Split images to train-test.ipynb notebook. It uses the same image splitting method as the previous notebook, but it has a few added features.

### <a id="3.0"></a>[3.0 Training and Detection.ipynb](https://github.com/Code-nano/Tensortflow_object_detection-01/blob/b31adc89934316b3a18d8f8ec942389d5d54dc02/3.0%20Training%20and%20Detection.ipynb)
This notebook trains an object detection model using TensorFlow Object Detection API. It includes evaluation, real-time object detection from webcam feed, and model export for use in web and mobile applications.

#### Features

- Setup paths and environment variables
- Download TensorFlow Models Pretrained Models from the Model Zoo
- Install TensorFlow Object Detection (TFOD) API
- Create a label map for the dataset
- Generate TensorFlow records for training and evaluation
- Copy the model configuration to the training folder
- Update the model configuration for transfer learning
- Train the object detection model
- Evaluate the trained model
- Load the trained model from a checkpoint
- Perform object detection on a single image
- Perform real-time object detection using the webcam
- Freeze the trained model graph
- Convert the model to TensorFlow.js format for use in web applications
- Convert the model to TensorFlow Lite format for use in mobile applications
- Zip and export the trained model

## Contributions
If you have any suggestions or improvements, feel free to create a pull request or open an issue.

## <a id="0.0"></a>Acknowledgments and License Information
I'd like to thank the open-source community for providing the resources and tools for this project. I'd also like to thank [Nicholas Renotte](https://github.com/nicknochnack) for the foundational knowledge and basic techniques used for object detection.

This project uses the following open-source libraries:

- [labelImg](https://github.com/heartexlabs/labelImg.git) (MIT License)
- [TensorFlow](https://github.com/tensorflow/models.git) (Apache 2.0 License)
- [OpenCV](https://github.com/opencv/opencv.git) (Apache 2.0 License)

Please see the [LICENSE](LICENSE) file for the full text of each license.
