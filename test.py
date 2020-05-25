#!/usr/bin/env python
# coding: utf-8

# In[4]:
import pickle
import TestFeatureExtraction
import numpy as np
import pandas as pd
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# In[ ]:

test_file_name = sys.argv[1]
test_file_features = TestFeatureExtraction.featureExtraction(test_file_name)
mlp_clf = pickle.load(open('mlp_clf_model.pkl', 'rb'))
y_pred = mlp_clf.predict(test_file_features)
print('Saved the output of MLP Classifier prediction in a csv file')
mlp_dataframe = pd.DataFrame(y_pred, columns=['Meal/NoMeal'])
mlp_dataframe.to_csv('MLPClassifier_output.csv')





