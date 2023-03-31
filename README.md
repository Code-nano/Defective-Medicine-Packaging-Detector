# TensorFlow Object Detection Repository

Welcome to my TensorFlow Object Detection Repository! This repository contains 5 Jupyter Notebook files (.ipynb) that I've made, modified, and used to implement object detection using TensorFlow.

## Table of Contents
- 0.0 Testing GPU.ipynb
- 1.0 Image collection.ipynb
- 2.0 Split images to train-test.ipynb
- 2.1 Split images (alternative).ipynb
- 3.0 Training and Detection.ipynb

## Getting Started
To get started, clone this repository and install the required dependencies. Follow the instructions in each notebook to execute the code.

bash
## Clone the repository
git clone <repository_url>


## Dependencies
Make sure you have the following dependencies installed:

i used an anconda enviroment 


Notebooks
0.0 Testing GPU.ipynb
This notebook aims to display the current versions of various Python libraries and check for GPU availability. Specifically, it reports the versions of TensorFlow, Keras, Python, Pandas, and Scikit-Learn, and verifies if a GPU device is available for TensorFlow.

1.0 Image collection.ipynb
This notebook sets up a real-time image capture and processing pipeline using OpenCV. It captures images from a webcam, crops and resizes them, and saves the resulting images to disk with unique names. The pipeline also provides a countdown timer before each image is captured, which allows users to prepare for the photo. This notebook also installs and compiles the LabelImg tool, which is a graphical image annotation tool used for labeling object bounding boxes in images. It helps to create the required training data for object detection models.

2.0 Split images to train-test.ipynb
This notebook splits a dataset of labeled images into training and testing sets, following a specified split ratio (default is 80% for training and 20% for testing). It copies the images and their corresponding XML annotation files to separate train and test folders. The XML files are used to generate a CSV file containing the image filename and its corresponding bounding box coordinates. This CSV file is used to generate a TFRecord file, which is the required format for training object detection models.

2.1 split images (alternative).ipynb
This is the improved version of the 2.0 Split images to train-test.ipynb notebook. It uses the same image splitting method as the previous notebook, but it has a few added features.

3.0 Training and Detection.ipynb
This notebook trains an object detection model using TensorFlow Object Detection API. It includes evaluation, real-time object detection from webcam feed, and model export for use in web and mobile applications.
Features
    Setup paths and environment variables
    Download TensorFlow Models Pretrained Models from the Model Zoo
    Install TensorFlow Object Detection (TFOD) API
    Create a label map for the dataset
    Generate TensorFlow records for training and evaluation
    Copy the model configuration to the training folder
    Update the model configuration for transfer learning
    Train the object detection model
    Evaluate the trained model
    Load the trained model from a checkpoint
    Perform object detection on a single image
    Perform real-time object detection using the webcam
    Freeze the trained model graph
    Convert the model to TensorFlow.js format for use in web applications
    Convert the model to TensorFlow Lite format for use in mobile applications
    Zip and export the trained model

Contributing
We welcome contributions to this repository. If you have any suggestions or improvements, feel free to create a pull request or open an issue.

License
This repository is licensed under the MIT License.

Acknowledgments
I'd like to thank the open-source community for providing the resources and tools for this project.
