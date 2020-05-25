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
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def featureExtraction(test_file_name):

    # loading the csv files
    dataset = pd.DataFrame()
    df = None

    with open(test_file_name, 'r') as csvfile:
        for line in csvfile:
            df = pd.read_csv(test_file_name, sep = '\t', header = None)
        dataset= dataset.append(df, ignore_index = True)

    # df = pd.read_csv(test_file_name)
    # dataset= dataset.append(df, ignore_index = True)

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

    feature_vector=[]
    for idx, row in dataset.iterrows():
        # cgm velocity
        cgm_velocity = [0]
        for index in range(len(row)-1):
            cgm_velocity += [row[index+1]-row[index]]
        cgm_velocity.sort(reverse=True)
        feature_vector += cgm_velocity[:4]

    feature_vector= np.array(feature_vector)
    feature_vector= np.reshape(feature_vector, (dataset.shape[0],4))

    feature_matrix = np.append(rolling_mean, rolling_std, axis = 1)
    feature_matrix = np.append(feature_matrix, Fourier_frequency, axis = 1)
    feature_matrix = np.append(feature_matrix, Fourier_peak, axis = 1)
    feature_matrix = np.append(feature_matrix, polyfit_reg, axis = 1)
    feature_matrix = np.append(feature_matrix, feature_vector, axis = 1)

    # Tackling the NAN values/missing values by replacing them with zeros
    where_are_NaNs = isnan(feature_matrix)
    feature_matrix[where_are_NaNs] = 0

    # create the covariance matrix
    sc = StandardScaler()
    X_std = sc.fit_transform(feature_matrix)
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    # print('Covariance matrix = \n%s' %cov_mat)

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
    var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)

    return pcaTransformed


# In[ ]:




