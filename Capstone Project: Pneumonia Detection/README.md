RSNA Pneumonia Detection Detection - Capstone Project
Capstone Project completed as a part of Great Learning's PGP - Artificial Intelligence and Machine Learning.

📁 Getting Started
The project is built on Google Colab Jupyter Notebook. Clone this repository to get started, as mentioned below. You can upload the cloned folder to your google drive or else git clone from google colab.

$ git clone https://github.com/sharmapratik88/Capstone_Pneumonia_Detection.git
🤔 Problem Statement
Computer vision can be used in health care for identifying  diseases. In pneumonia detection we need to detect inflammation of the lungs.   In  this   challenge,   you’re  required  to    build    an    algorithm    to   detect    a   visual    signal     for    pneumonia    in   medical   images. Specifically,  your  algorithm  needs  to automatically  locate  lung  opacities  on chest radiographs.

📜 Approach
📈 Step 1: Exploratory Data Analysis & Data Preparation
Understanding the data with a brief on train/test labels and respective class info
Look at the first five rows of both the csvs (train and test)
Identify how are classes and target distributed
Check the number of patients with 1, 2, ... bounding boxes
Read and extract metadata from dicom files
Perform analysis on some of the features from dicom files
Check some random images from the training dataset
Draw insights from the data at various stages of EDA
Visualize some random masks generated
Outcome

Jupyter Notebook Link containing the exploration steps.
Module Link contains custom module which was built to help in performing EDA.
Data Generator contains custom module which was built for data generate and help in visualizing the masks.
Output (pickle files) contains output files such as train_class_features.pkl containing metadata features and train_feature_engineered.pkl after feature engineering on training dataset.
⚙️ Step 2: Model Building
Split the data
Use DenseNet-121 architecture
Make use of pre-trained CheXNet weights and train the model
Evaluate the models (ROC AUC, AP, F1 Score)
Outcome

Classification - Jupyter Notebook Link with the DenseNet-121 architecture with pretrained ImageNet weights trained on RSNA Pneumonia Detection dataset. Evaluating the model on average precision, accuracy and ROC AUC. Also compared the DenseNet-121 output with the pretrained CheXNet weights.
Segmentation - Jupyter Notebook Link with the CNN based segmentation. Evaluating the model on IOU and validation loss.
Module Link contains custom module which was built to help in model building.
Output (pickle files) contains train, valid and test pickle files after split on training dataset.
Acknowledgments
We used pre-trained weights available from the following repository.
