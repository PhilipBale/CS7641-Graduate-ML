import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


from collections import Counter
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.base import TransformerMixin,BaseEstimator
import scipy.sparse as sps
from scipy.linalg import pinv

nn_layers = [(100,), (50,), (50, 50)]
nn_reg = [10**-x for x in range(1,5)]
nn_iter = 1500

clusters =  [2,5,10,15,20,25,30,35,40, 50]
dims = [2,3, 4, 5, 6, 7,] # 8, 9, 10,15,20,25,30,35,40,45,50,55,60]

def load_data():
  np.random.seed(0)


  perm = pd.read_csv("./data/perm.csv")
  perm_Y_VAL = 'case_status'

  housing = pd.read_csv('./data/housing.csv')
  housing_Y_VAL = 'price_bracket'


  perm_y = perm[perm_Y_VAL].copy().values
  perm_x = perm.drop(perm_Y_VAL, 1).copy().values

  housing_y = housing[housing_Y_VAL].copy().values
  housing_x = housing.drop(housing_Y_VAL, 1).copy().values


  perm_x = StandardScaler().fit_transform(perm_x)
  housing_x = StandardScaler().fit_transform(housing_x)

  return (perm_x, perm_y, housing_x, housing_y)


def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
#    assert max(pred) == max(Y)
#    assert min(pred) == min(Y)    
    return acc(Y,pred)


class myGMM(GMM):
    def transform(self,X):
        return self.predict_proba(X)