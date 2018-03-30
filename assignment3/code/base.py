
# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 6
HIDDEN_LAYER1 = 100
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 1000
OUTFILE = './NN_OUTPUT/BACKPROP_LOG.txt'


# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:17:14 2017

@author: JTay
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = './results/base/'
np.random.seed(0)


df1 = pd.read_csv("./data/perm.csv")
df1_Y_VAL = 'case_status'

df2 = pd.read_csv('./data/housing.csv')
df2_Y_VAL = 'price_bracket'


df1_y = df1[df1_Y_VAL].copy().values
df1_x = df1.drop(df1_Y_VAL, 1).copy().values

df2_y = df2[df2_Y_VAL].copy().values
df2_x = df2.drop(df2_Y_VAL, 1).copy().values



df1_x = StandardScaler().fit_transform(df1_x)
df2_x = StandardScaler().fit_transform(df2_x)

#%% benchmarking for chart type 2

grid ={'NN__alpha':[10**-x for x in range(1,5)],'NN__hidden_layer_sizes':[(100,), (1, 100), (50, 50)]}

mlp = MLPClassifier(activation='relu',max_iter=TRAINING_ITERATIONS,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(df1_x,df1_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'1.csv')


mlp = MLPClassifier(activation='relu',max_iter=TRAINING_ITERATIONS,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(df2_x,df2_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'2.csv')