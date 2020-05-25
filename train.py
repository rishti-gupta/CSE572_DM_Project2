#!/usr/bin/env python
# coding: utf-8

# In[635]:


# importing python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import csv
import pickle
# create expanding window features
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import warnings
warnings.filterwarnings('ignore')
from operator import is_not
from functools import partial
from sklearn import preprocessing
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# def featureExtraction(test_file_name):

# loading the csv files
# meal data
dataset = pd.DataFrame()
df = None
# label1 = list()
# label2 = list()

for i in range(5):
    df = pd.read_csv("MealNoMealData/mealData{}.csv".format(i+1), sep = '\t', header = None)
    dataset = dataset.append(df, ignore_index = True)

# no meal data
for i in range(5):
    df = pd.read_csv("MealNoMealData/noMeal{}.csv".format(i+1), sep = '\t', header = None)
    dataset = dataset.append(df, ignore_index = True)

dataset= dataset[0].str.split(',', expand = True)
# dataset = dataset.dropna(how = 'all')
dataset = dataset.fillna(0)
dataset = dataset.replace("NaN", 0)
dataset = dataset.astype(float)
dataset_arr = np.array(dataset)

# rolling mean and rolling standard deviation
data = dataset.T
rolling_mean = data.rolling(window=3,min_periods=3).mean()
rolling_mean = rolling_mean.T
rolling_std = data.rolling(window=3,min_periods=3).std()
rolling_std = rolling_std.T

# fft feature
cgmFFTValues = abs(np.fft.fft(dataset_arr))
freq = np.fft.fftfreq(dataset_arr.shape[-1])

# picking top 8 peaks of FFT
FFT=np.array(cgmFFTValues)
fft_freq=np.array(freq)
Fourier_peak=list()
Fourier_frequency=list()
for i in range(len(FFT)):
    index=np.argsort(FFT)[i][-9:]

    peak=FFT[i][index]
    Fourier_peak.append(peak)
    freq=abs(fft_freq[index])
    freq.sort()
    fr=freq[[0,1,3,5,7]]
    Fourier_frequency.append(fr)

Fourier_peak=np.array(Fourier_peak)
Fourier_frequency=np.array(Fourier_frequency)
Fourier_peak=np.unique(Fourier_peak,axis=1)

# polyfit regression feature
polyfit_reg = []
x = [i for i in range(dataset.shape[1])]

for i in range(len(dataset_arr)):
    polyfit_reg.append(np.polyfit(x, dataset_arr[i], 3))

polyfit_reg = np.array(polyfit_reg)

# cgm velocity feature
feature_vector=[]
for idx, row in dataset.iterrows():
    # cgm velocity
    cgm_velocity = [0]
    for index in range(len(row)-1):
        cgm_velocity += [row[index+1]-row[index]]
    cgm_velocity.sort(reverse=True)
    feature_vector += cgm_velocity[:5]

feature_vector= np.array(feature_vector)
feature_vector= np.reshape(feature_vector, (dataset.shape[0],5))
# print feature_vector

# zc = []
# for i in range(510):
#   zero_crossings = np.where(np.diff(np.sign(feature_vector[i])))[0]
#   zc.append(len(zero_crossings))
#
# zc = np.array(zc)
# zc = np.reshape(zc, (dataset.shape[0], 1))

# creating the feature matrix
feature_matrix = np.append(rolling_mean, rolling_std, axis = 1)
feature_matrix = np.append(feature_matrix, Fourier_frequency, axis = 1)
feature_matrix = np.append(feature_matrix, Fourier_peak, axis = 1)
feature_matrix = np.append(feature_matrix, polyfit_reg, axis = 1)
feature_matrix = np.append(feature_matrix, feature_vector, axis = 1)
# feature_matrix = np.append(feature_matrix, zc, axis = 1)

# Tackling the NAN values/missing values by replacing them with zeros

where_are_NaNs = isnan(feature_matrix)
feature_matrix[where_are_NaNs] = 0

# create the covariance matrix
sc = StandardScaler()
X_std = sc.fit_transform(feature_matrix)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

# create eigen values and eigen vectors
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# feed the feature matrix to PCA
feature_matrix = StandardScaler().fit_transform(feature_matrix)
df_feature_matrix = pd.DataFrame(feature_matrix)
df_feature_matrix.fillna(0, inplace=True)

data_scaled = pd.DataFrame(preprocessing.scale(df_feature_matrix), columns=df_feature_matrix.columns)
pca = decomposition.PCA(n_components=20)
X_std_pca = pca.fit_transform(data_scaled)

# calculate the explained variance of pca
pcaExpVariance = pca.explained_variance_
# print("PCA variance= ", pcaExpVariance)
pcaTransformed = pca.transform(feature_matrix)

# calculate explained variance ratio for analysis of no. of features
# using 4 components
variance = pca.explained_variance_ratio_
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)

dataset = pd.DataFrame(pcaTransformed)
k=20
dataset['target'] = [1] * 255 + [0] * 255
# randomize
dataset = dataset.sample(frac=1).reset_index(drop=True)
# dataset = np.array(dataset)

X = dataset.iloc[:, 0:k]
# X = dataset
y = dataset.target

accuracy, f1, precision, recall = [], [], [], []

# k-fold cross validation
k_fold = KFold(n_splits = 10, shuffle = True)
np.random.seed(1)
mlp_classifier = MLPClassifier(hidden_layer_sizes=100, activation='logistic', max_iter=2000, alpha=0.00001,
                      learning_rate='adaptive', solver='adam', tol=0.000002)
# svm_classifier = LinearSVC()
# rf_classifier = RandomForestClassifier(max_depth=2, random_state=0)
# gNB_classifier = GaussianNB()
# xGB_classifier = XGBClassifier()

for train_index, test_index in k_fold.split(X):
    Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
    ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

    mlp_classifier.fit(Xtrain, ytrain)
    # svm_classifier.fit(Xtrain, ytrain)
    # rf_classifier.fit(Xtrain, ytrain)
    # gNB_classifier.fit(Xtrain, ytrain)
    # xGB_classifier.fit(Xtrain, ytrain)

    mlp_predicted = mlp_classifier.predict(Xtest)
    predictions = [round(value) for value in mlp_predicted]
    # svm_predicted = svm_classifier.predict(Xtest)
    # rf_predicted = rf_classifier.predict(Xtest)
    # gNB_predicted = gNB_classifier.predict(Xtest)
    # xGB_classifier = gNB_classifier.predict(Xtest)

    accuracy.append(accuracy_score(predictions, ytest))
    f1.append(f1_score(predictions, ytest))
    precision.append(precision_score(predictions, ytest))
    recall.append(recall_score(predictions, ytest))

    # accuracy1.append(accuracy_score(ytest, svm_predicted))
    # f11.append(f1_score(ytest, svm_predicted))
    # precision1.append(precision_score(ytest, svm_predicted))
    # recall1.append(recall_score(ytest, svm_predicted))
    #
    # accuracy2.append(accuracy_score(ytest, rf_predicted))
    # f12.append(f1_score(ytest, rf_predicted))
    # precision2.append(precision_score(ytest, rf_predicted))
    # recall2.append(recall_score(ytest, rf_predicted))
    #
    # accuracy3.append(accuracy_score(ytest, gNB_predicted))
    # f13.append(f1_score(ytest, gNB_predicted))
    # precision3.append(precision_score(ytest, gNB_predicted))
    # recall3.append(recall_score(ytest, gNB_predicted))

    # accuracy4.append(accuracy_score(ytest, gNB_predicted))
    # f14.append(f1_score(ytest, gNB_predicted))
    # precision4.append(precision_score(ytest, gNB_predicted))
    # recall4.append(recall_score(ytest, gNB_predicted))

filename = 'mlp_clf_model.pkl'
outfile = open(filename, 'wb')
pickle.dump(mlp_classifier, outfile)
outfile.close()

print("\nUsing MLP Classifier: \n")
print("Accuracy: %.2f%%" % (np.mean(accuracy) * 100.0))
print("F1 Score : %.2f%%" %  (np.mean(f1)* 100.0))
print("Precision Score: %.2f%%" % (np.mean(precision)* 100.0))
print("Recall Score: %.2f%%" % (np.mean(recall)* 100.0))


# In[ ]:






