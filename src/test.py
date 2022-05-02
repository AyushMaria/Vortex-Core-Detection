# -*- coding: utf-8 -*-
"""Test.ipynb

This file calls the Voting Classifier that is trained using the training dataset, and 
implements it to predict the vortex cores. 

Original file is located at
    https://github.com/AyushMaria/Vortex-Detection
"""

# importing linear algebra
import numpy as np 

# importing tools for data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# importing the tools to call google drive
from google.colab import drive

# importing seaborn
import seaborn as sns

from sklearn.model_selection import GridSearchCV

# import train_test_split function
from sklearn.model_selection import train_test_split

# import the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Import the AdaBoost classifier
from sklearn.ensemble import AdaBoostClassifier

# Import the XGBoost Classifier 
from xgboost import XGBClassifier

# Import joblib to save machine learning model
import joblib

# Import the Voting CLassifier
from sklearn.ensemble import VotingClassifier

# Importing the necessary tools to check how the clasifiers performed
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 

# import the required metrics to plot validation curve
from sklearn.model_selection import validation_curve

# import the plotting library to visualize our learning curve
import matplotlib.pyplot as plt

# Load the model from the file
votingclassifier_from_joblib = joblib.load('Voting Classifier.pkl')

def predict():
"""Predict Function

This function calls the test dataset. It then calls the Voting Classifier and predicts the cores of the vortices.
It further writes the predicted cores to predictions.csv and saves them  

Original file is located at
    https://github.com/AyushMaria/Vortex-Detection
"""	

	df = pd.read_csv('features.csv')
	df=df.loc[df.contourArea>100]

	# Importing the image feature dataset for the classifier
	final_X_test=df[['contourArea','areaPercDiff','aspectRatio','momentLocDiff']]
	final_y_pred=votingclassifier_from_joblib.predict(final_X_test)

	# Creating a data frame file for the Predicted Core Locations
	row=[]
	predictions=final_X_test.copy()
	predictions['Z']=df['Z']
	predictions['Y']=df['Y']
	predictions['label']=final_y_pred
	predictions['ID']=df['ID']
	predictions=predictions.loc[predictions.label=='vortex']
	a=range(predictions[predictions.columns[0]].count())
	row=[x+1 for x in a]
	predictions['row']=row
	
	# Arranging the columns in the desired manner
	predictions=predictions[['row','ID','areaPercDiff','aspectRatio','momentLocDiff','label','Z','Y']]

	# Saving the Predictions Data Frame as predictions.csv
	predictions.to_csv('predictions.csv')

predict()