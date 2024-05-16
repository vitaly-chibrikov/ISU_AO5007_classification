#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:14:06 2024

@author: vitaly
"""
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def csv_reader(path, delimeter):
    X = pd.read_csv(path, sep=delimeter)
    return X

def column_filter(X, column_names):
    X=X.drop(labels=column_names, axis=1)
    return X
    
def missing_values(X):
    X=X.fillna("missing")
    return X

def row_filter(X, column_to_test, pattern, filter_type):
    if filter_type == "exclude":
        column=X[column_to_test]
        indexes=X[column == pattern].index
        X=X.drop(indexes)
    return X

def replace_rare_values(X_col, threshold):
    series = pd.Series(list(X_col))
    values_count=series.value_counts()
    for i, v in values_count.items():
        if v < threshold:
            X_col.mask(X_col==i, other="rare_value", inplace=True)

def print_series(X, show_count=False):
    print("*********************** Uniques ***********************")
    
    for col in X:
        print(col, ": ", pd.Series(list(X[col])).unique())
      
    print("*********************** Values counts ***********************")

    if show_count:
        for col in X:
            series = pd.Series(list(X[col]))
            print("Variable: ", col)
            print(series.value_counts())
            print("Size: ", series.value_counts().size)
            print()
    
def plot_kmeans(X, iterations):
    inertias = []

    for i in range(1,iterations):
        kmeans = KMeans(n_clusters=i, verbose=False)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1,iterations), inertias, marker='o')
    plt.title('KMeans inertias')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def check_outliers(X):
    clf=LocalOutlierFactor()
    outlier=clf.fit_predict(X)
    print("Otliers found:", (outlier == -1).sum())
    return outlier

def add_SMOTE(X, y):
    #Synthetic Minority Over-sampling Technique 
    print("y balance before SMOTE: ")
    print(pd.Series(list(y)).value_counts())
    over_sampling=SMOTE()
    X,y=over_sampling.fit_resample(X, y)
    print("y balance after SMOTE: ")
    print(pd.Series(list(y)).value_counts())
    print("X shape after balancing: ", X.shape)
    print("y shape after balancing: ", y.shape)
    return X,y

def undersample(X, category_name):
    y=X[category_name]
    
    X_y_1=X[y == 1]
    X_y_0=X[y == 0]

    X_y_1=X_y_1.sample(n=X_y_0.shape[0], random_state=1)
    
    X=pd.concat([X_y_1, X_y_0])
    return X

def csv_writer(X, path):
    X.to_csv(path, index=False, sep=';')
    
def one_to_many(X):
    X=pd.get_dummies(X,dtype=int)
    return X

def remove_outliers(X, class_name):
    y=X[class_name]
    X=X.drop(labels=class_name, axis=1)
    clf=LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    outlier=clf.fit_predict(X)
    print("Otliers found:", (outlier == -1).sum())
    #Remove outliers
    X=X[outlier == 1]
    y=y[outlier == 1]
    X[class_name]=y
    return X