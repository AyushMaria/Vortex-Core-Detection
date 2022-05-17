# Vortex Core Detection
Vortex Cores Detection Using Bagging and Boosting Algorithms 

Yolo V5, an object detection algorithm, is utilized to find vortices in the image dataset that contains vortex images. Furthermore, feature extraction methods are deployed to extract features that are the most relevant to finding vortex cores. 

Data Wrangling, Exploratory Data Analysis and Feature engineering are then used to highlight trends that are important within the feature dataset. Outliers are discarded and the dataset is cleaned.

Finally the cleaned data is passed into Machine Learning Ensembles like AdaBoost Algorithm, Random Forest Classifier etc to predict the location of the cores. The final ensemble created is a voting classifier with a hard voting feature which consists of AdaBoost Algorithm, Random Forest Classifier and XGBoost Classifier.

The whole process is implemented on Google VM in the backend along with a sophisticated front end which was built in HTML.
