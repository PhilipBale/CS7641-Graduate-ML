import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters, dims
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

out = './results/pca/'

perm_x, perm_y, housing_x, housing_y = load_data() # perm, housing


# 2

pca = PCA(random_state=5)
pca.fit(perm_x)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,501))
tmp.to_csv(out+'perm scree.csv')


pca = PCA(random_state=5)
pca.fit(housing_x)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,65))
tmp.to_csv(out+'housing scree.csv')


#3 TODO

#4

grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(perm_x,perm_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'perm dim red.csv')


grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(housing_x,housing_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'housing dim red.csv')

# #%% data for 3
# # Set this from chart 2 and dump, use clustering script to finish up
# dim = 5
# pca = PCA(n_components=dim,random_state=10)

# perm_x2 = pca.fit_transform(perm_x)
# perm2 = pd.DataFrame(np.hstack((perm_x2,np.atleast_2d(perm_y).T)))
# cols = list(range(perm2.shape[1]))
# cols[-1] = 'Class'
# perm2.columns = cols
# perm2.to_hdf(out+'datasets.hdf','perm',complib='blosc',complevel=9)

# dim = 60
# pca = PCA(n_components=dim,random_state=10)
# housing_x2 = pca.fit_transform(housing_x)
# housing2 = pd.DataFrame(np.hstack((housing_x2,np.atleast_2d(housing_y).T)))
# cols = list(range(housing2.shape[1]))
# cols[-1] = 'Class'
# housing2.columns = cols
# housing2.to_hdf(out+'datasets.hdf','housing',complib='blosc',complevel=9)