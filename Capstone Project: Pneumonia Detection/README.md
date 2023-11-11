**RSNA Pneumonia Detection Detection - Capstone Project**
Capstone Project completed as a part of IIIT Delhi's Post Graduate Diploma in Artificial Intelligence


**Problem Statement**
Pneumonia is a prevalent respiratory disease that requires early detection and treatment.The goal of this project is to develop a deep learning algorithm for detecting lung opacity in chest X-Ray images. Computer vision techniques can be applied to chest radiographs to automatically identify lung opacities, a critical indicator of pneumonia.

**Approach**
Phase 1: Data Preprocessing and Exploration
-	Loading and preprocessing DICOM images using Python libraries (pydicom, cv2).
-	Exploring the dataset to understand its structure, distribution, and labeling.
-	Identify how are classes and target distributed
-	Splitting the dataset into training, validation, and test sets.
Phase 2: Model Development
-	Using transfer learning architectures suitable for image classification.
-	Training the model on the training dataset with lung opacity detection as the primary objective.
=	Implementing data augmentation techniques to enhance model generalization.
Phase 3: Model Evaluation and Refinement
-	Validating the model on the validation dataset to assess its performance.
-	Analyzing evaluation metrics and adjust hyperparameters as needed.
-	Fine-tune the model to achieve the best possible accuracy and generalization.
Phase 4: Result and Conclusion
-	summarizing the project's objectives, methodology, results, and conclusions.
-	comparing the performance metrics of the different transfer learning models with and without using image augmentation.


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
