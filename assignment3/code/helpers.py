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
from collections import defaultdict
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from time import clock
from sklearn.metrics import adjusted_mutual_info_score as ami


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


def run_clustering(out, perm_x, perm_y, housing_x, housing_y):
  SSE = defaultdict(dict)
  ll = defaultdict(dict)
  acc = defaultdict(lambda: defaultdict(dict))
  adjMI = defaultdict(lambda: defaultdict(dict))
  km = kmeans(random_state=5)
  gmm = GMM(random_state=5)

  st = clock()
  for k in clusters:
      km.set_params(n_clusters=k)
      gmm.set_params(n_components=k)
      km.fit(perm_x)
      gmm.fit(perm_x)
      SSE[k]['perm'] = km.score(perm_x)
      ll[k]['perm'] = gmm.score(perm_x)    
      acc[k]['perm']['Kmeans'] = cluster_acc(perm_y,km.predict(perm_x))
      acc[k]['perm']['GMM'] = cluster_acc(perm_y,gmm.predict(perm_x))
      adjMI[k]['perm']['Kmeans'] = ami(perm_y,km.predict(perm_x))
      adjMI[k]['perm']['GMM'] = ami(perm_y,gmm.predict(perm_x))
      
      km.fit(housing_x)
      gmm.fit(housing_x)
      SSE[k]['housing'] = km.score(housing_x)
      ll[k]['housing'] = gmm.score(housing_x)
      acc[k]['housing']['Kmeans'] = cluster_acc(housing_y,km.predict(housing_x))
      acc[k]['housing']['GMM'] = cluster_acc(housing_y,gmm.predict(housing_x))
      adjMI[k]['housing']['Kmeans'] = ami(housing_y,km.predict(housing_x))
      adjMI[k]['housing']['GMM'] = ami(housing_y,gmm.predict(housing_x))
      print(k, clock()-st)
      
      
  SSE = (-pd.DataFrame(SSE)).T
  SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
  ll = pd.DataFrame(ll).T
  ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
  acc = pd.Panel(acc)
  adjMI = pd.Panel(adjMI)


  SSE.to_csv(out+'SSE.csv')
  ll.to_csv(out+'logliklihood.csv')
  acc.ix[:,:,'housing'].to_csv(out+'Housing acc.csv')
  acc.ix[:,:,'perm'].to_csv(out+'Perm acc.csv')
  adjMI.ix[:,:,'housing'].to_csv(out+'Housing adjMI.csv')
  adjMI.ix[:,:,'perm'].to_csv(out+'Perm adjMI.csv')