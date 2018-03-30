
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters, dims, run_clustering
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import pandas as pd

out = './results/pca/'

perm_x, perm_y, housing_x, housing_y = load_data() # perm, housing


# 2

pca = PCA(random_state=5)
pca.fit(perm_x)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,8))
tmp.to_csv(out+'perm scree.csv')


pca = PCA(random_state=5)
pca.fit(housing_x)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,13))
tmp.to_csv(out+'housing scree.csv')

#4

grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(perm_x,perm_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'perm dim red.csv')


grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(housing_x,housing_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'housing dim red.csv')

# 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 7
pca = PCA(n_components=dim,random_state=10)
perm_x2 = pca.fit_transform(perm_x)

dim = 12
pca = PCA(n_components=dim,random_state=10)
housing_x2 = pca.fit_transform(housing_x)

run_clustering(out, perm_x2, perm_y, housing_x2, housing_y)
