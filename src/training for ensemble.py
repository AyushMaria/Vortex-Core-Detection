# -*- coding: utf-8 -*-
"""Train.ipynb

Original file is located at https://github.com/AyushMaria/Vortex-Detection
"""

# importing linear algebra
import numpy as np 

# importing tools for data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# import directory
import os
from os import chdir
import sys 
from pathlib import Path


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
from sklearn.preprocessing import LabelEncoder

# Importing the necessary tools to check how the clasifiers performed
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn import metrics

# import the required metrics to plot validation curve
from sklearn.model_selection import validation_curve

# import the plotting library to visualize our learning curve
import matplotlib.pyplot as plt



def hyperparameter_tune(X_train,y_train):
    '''This function uses grid search library from sci-kit learn to 
    run every parameter mentioned within on the classifiers selected. 
    
    The parameters that provide the best output are then returned as a 
    recommendation to tweak the models in the function training'''
    xg_param = {
        "n_estimators": [50, 100],
        "random_state": [0,1],
        "max_depth" : [2,3],
        "learning_rate" :[0.5,0.7]
    }
    rf_param = {
        "n_estimators": [50, 100],
        "random_state": [0,1],
        "max_depth" : [2,3]
    }

    abc_param = {
        "n_estimators": [50, 100],
        "random_state": [0,1],
     "learning_rate" :[0.5,0.7]
    }


    params = {}
    params.update({"xg__" + k: v for k, v in xg_param.items()})
    params.update({"rf__" + k: v for k, v in rf_param.items()})
    params.update({"abc__" + k: v for k, v in abc_param.items()})

    xg = XGBClassifier()
    rf = RandomForestClassifier()
    abc= AdaBoostClassifier()
    eclf = VotingClassifier(estimators=[("xg", xg),("rf", rf),("abc", abc)],voting="hard")


    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=2)

    grid.fit(X_train,y_train)
    print (grid.best_params_)
    return grid.best_params_

def train_classifiers(X_train,X_test,y_train,y_test):
    '''This function trains the classifiers and prints the validation accuracy.
    It furhter calls feature importance function and the confusion matrix function
    and saves them in the folder images.
    
    It also calls the classification report function.
    
    It saves the machine learning model to the directory as a pkl file '''


    # Creating classifer objects and assigning them with the appropriate parameters
    clf1 = AdaBoostClassifier(n_estimators=50, learning_rate=0.5, random_state=0)
    clf2 = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0)
    clf3 = XGBClassifier(n_estimators=50, learning_rate=0.5, random_state=0, max_depth=2)

    # Training the Classifers using the fit method
    model1 = clf1.fit(X_train, y_train)
    model2 = clf2.fit(X_train, y_train)
    model3 = clf3.fit(X_train, y_train)
    
    # Making a folder to save all images in
    os.makedirs('images', exist_ok=True)

    # Calling the feature importance function for each model
    feature_importance(model1)
    feature_importance(model2)
    feature_importance(model3)
    
    
    #validation_curve(model1,X_train,y_train)

    # Assigning the fitted models to the voting classifier and then fitting the voting classifier
    eclf = VotingClassifier( estimators=[('abc', model1), ('rf', model2), ('xgb', model3)],voting='hard')
    model4 = eclf.fit(X_train, y_train)

    # Determining the accuracy of each model 
    y_pred = model4.predict(X_test)
    print(model4, accuracy_score(y_test, y_pred))

    classification_report(y_test, y_pred)
    confusion_matrix(y_test,y_pred)

    # Save the model as a pickle in a file
    joblib.dump(model4, 'Voting Classifier.pkl')


def classification_report(y_test, y_pred):
    '''This function prints the classification report of the model's performance'''
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    os.makedirs('reports', exist_ok=True) 
    df_classification_report.to_csv('reports/report.csv')


def validation_curve(model,X,y):
    '''This function plots the validation curves of the classifiers.'''
    # arranging the parameters that have to be tested
    param_range = np.arange(1, 51)

    # obtaining the training and the testing scores to plot on the graph

    train_score, test_score = validation_curve(model, X, y, param_name = "n_estimators", param_range=param_range, cv = 5, scoring = "accuracy")
 
    # Calculating the mean and the standard deviation of the training score
    mean_training_score = np.mean(train_score, axis = 1)
    std_training_score = np.std(train_score, axis = 1)
 
    # Calculating the mean and the standard deviation of the testing score
    mean_testing_score = np.mean(test_score, axis = 1)
    std_testing_score = np.std(test_score, axis = 1)
 
    # Plot the mean accuracy scores for the training and testing scores
    plt.plot(param_range, mean_training_score, label = "Training Score", color = 'b')
    plt.plot(param_range, mean_testing_score, label = "Cross Validation Score", color = 'g')
 
    # Creating the plot
    plt.title("Validation Curve with AdaBoost Classifier")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.savefig('curve.png')

def confusion_matrix(y_test,y_pred):
    '''
    This function pronts the confusion matrix as a heatmap and saves it to the pc in the images folder.'''
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    #group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(5,5), dpi= 80)
    sns_plot=sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    fig = sns_plot.get_figure()
    fig.savefig("images/heatmap.png")

def feature_importance(clf):
    '''This function saves the feature importance of each algorithm in the images folder'''
    importance = clf.feature_importances_
    plt.figure(figsize=(5,5), dpi= 80)
    plt.bar([x for x in range(len(importance))], importance)
    plt.savefig('images/feature_importance.png')


def main():
    '''
    The main function calls the trairning dataset and splits it into the 
    training and validation formats for training purposes. It serves as the
    main function that carries out all tasks.
    '''
    # Setting the directory to the input files
    os.chdir(".\\data")

    # Importing the image feature dataset for the classifier
    data = pd.read_csv('train.csv')

    # dropping the rows with null values
    train=data.dropna()

    # creating a column which consists of row numbers 
    train.insert(loc=0, column='row_num', value=np.arange(len(train)))

    # dropping irrelevant features
    train=train.drop(columns=['ID','localX', 'localY'])

    # predictions = pd.DataFrame(columns = ['contourArea','areaPercDiff','aspectRatio','momentLocDiff'])

    # Assigning the important features to X 
    X = train[['contourArea','areaPercDiff','aspectRatio','momentLocDiff']]

    # Assigning the label values to y
    y = train['label']

    le=LabelEncoder()

    y=le.fit_transform(y)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    train_classifiers(X_train,X_test,y_train,y_test)

main()