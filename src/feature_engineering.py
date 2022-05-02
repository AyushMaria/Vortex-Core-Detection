# -*- coding: utf-8 -*-
"""Feature Engineering.ipynb

Original file is located at https://github.com/AyushMaria/Vortex-Detection
"""
import os
import plotly
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as font_manager
from pandas.plotting import scatter_matrix
from datetime import datetime,timedelta
import numpy as np

def plot_trends(df):
    '''
    This function plots different plots for the dataset obtained.
    The plots that have featured in this function are 
    Count Plot 
    Density Plot
    Pair Plot
    Box Plot
    Point Plot
    Strip Plot
    Violin Plot
    Joint Plot
    Heatmap'''
    y=df['label']
    X = df[['contourArea','areaPercDiff','aspectRatio','momentLocDiff']]
    ax=sns.countplot(y,label='count', palette='crest')
    fig = ax.get_figure()
    fig.savefig("images/countplot.png") 
    V,NV = y.value_counts()
    print('Vortex: ',V)
    print('Non - Vortex: ',NV)

    plt.figure(figsize=(5,5), dpi= 80)
    sns.kdeplot(data=df, x="areaPercDiff", hue="label", fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)
    plt.title('Density Plot of Vortex Values by Area Percentage Difference', fontsize=15)
    plt.xlabel("Area Percentage Difference",fontsize=15)
    #plt.legend()
    plt.savefig('images/densityplot.png')

    #X['label']=df['label'].copy()

    plt.figure(figsize=(15,10), dpi= 80)
    g = sns.PairGrid(df.drop(columns=['row_num','Z','Y']), hue='label',palette='crest')
    g.map_diag(sns.scatterplot)
    g.map_offdiag(sns.kdeplot, fill=True)
    g.savefig("images/pairplot1.png")

    g=sns.pairplot(df.drop(columns=['row_num','Z','Y']), hue="label", palette='crest')
    g.fig.set_size_inches(10,10)
    g.savefig("images/pairplot2.png")
    
    
    plt.figure(figsize=(15,10), dpi= 80)
    x_vars = ["areaPercDiff", "aspectRatio", "momentLocDiff", "contourArea"]
    y_vars = ["areaPercDiff"]
    g = sns.PairGrid(df.drop(columns=['row_num','Z','Y']), hue='label', x_vars=x_vars, y_vars=y_vars, palette='crest')
    g.map_diag(sns.kdeplot, fill=True)
    g.map_offdiag(sns.kdeplot, fill=True)
    g.savefig("images/pairplot3.png")

    #X=X.drop(columns=['label'])
    M=X['areaPercDiff']

    data_dia = y
    data = X
    data_n_2 = (data - data.mean()) / (data.std())             
    data = pd.concat([y,data_n_2.iloc[:,]],axis=1)
    data = pd.melt(data,id_vars="label",var_name="features", value_name='value')

    fig, ax = plt.subplots()
    g=sns.boxplot(x="features", y="value", hue="label", data=data, palette='crest', ax=ax)
    plt.xticks(rotation=45)
    plt.savefig("images/boxplot.png")

    fig, ax = plt.subplots()
    g = sns.pointplot(x="features", y="value", hue="label", data=data, palette="PuBu",ax=ax)
    plt.savefig("images/pointplot.png")
    fig, ax = plt.subplots()
    g = sns.violinplot(x="features", y="value", hue="label", data=data,split=True, inner="quart", palette='PuBu',ax=ax)
    plt.xticks(rotation=45)
    plt.savefig("images/violinplot.png")


    fig, ax = plt.subplots()
    g = sns.jointplot(X.loc[:,'areaPercDiff'],X.loc[:,'aspectRatio'],kind="reg",ax=ax)
    plt.savefig("images/jointplot.png")


    fig, ax = plt.subplots()
    g= sns.stripplot(x="features", y="value", hue="label", data=data, marker="D", size=10, edgecolor="gray", alpha=.25,ax=ax)
    plt.xticks(rotation=45)
    plt.savefig("images/stripplot.png")


    fig,ax = plt.subplots(figsize=(5,5))
    cmap = sns.light_palette("#0a6ba4", as_cmap=True)
    g = sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap=cmap);
    plt.savefig("images/heatmap.png")


def main():
    '''
    This is the main function.
    It calls the dataset and sends it to the plotting function.
    It further preprocesses the dataset for plotting needs'''
    # Setting the directory to the input files
    os.chdir(".\\data")

    df=pd.read_csv('train.csv')
    df=df.dropna()
    df.insert(loc=0, column='row_num', value=np.arange(len(df)))
    df=df.drop(columns=['ID','localX', 'localY'])
    plot_trends(df)

main()