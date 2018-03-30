import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from random import sample
from collections import defaultdict
import graphviz 
import pandas


df = pd.read_csv('./perm_base.csv')
df = df[['naics_2007_us_code', 'wage_offer_from_9089', 'case_status', 'wage_offer_unit_of_pay_9089', 'country_of_citzenship', 'employer_state', 'pw_level_9089']]

df_x = [
    'wage_offer_from_9089',
    'naics_2007_us_code', 'wage_offer_unit_of_pay_9089', 'country_of_citzenship', 'employer_state', 'employer_state', 'pw_level_9089']
df_y = 'case_status'
to_encode = [ 'country_of_citzenship', 'employer_state', 'pw_level_9089']
 
    
le = preprocessing.LabelEncoder
encoderDict = defaultdict(le)

for column in to_encode:
    df[column] = df[column].dropna()
    df = df[df[column].notnull()]
    df[column] = encoderDict[column].fit_transform(df[column])


df = df[~df['case_status'].str.contains("Withdrawn", na=False)]
df.loc[(df['case_status'].str.contains('Certified', na=False)), 'case_status'] = 1
df.loc[(df['case_status'].str.contains('Denied', na=False)), 'case_status'] = -1

df.loc[(df['wage_offer_unit_of_pay_9089'].str.contains('yr', na=False)), 'wage_offer_unit_of_pay_9089'] = 2
df.loc[(df['wage_offer_unit_of_pay_9089'].str.contains('hr', na=False)), 'wage_offer_unit_of_pay_9089'] = 1

    
df = df.apply(lambda x: pandas.to_numeric(x.astype(str).str.replace(',',''), errors='coerce'))

df = df.dropna()  
# sample = df.sample(n=10000)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('perm.csv')









# Housing







df = pd.read_csv('./housing_base.csv')
housing_columns = ['MSSubClass', 'LotArea', 'Neighborhood', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'ExterCond', 'CentralAir', 'YrSold']
df = df[housing_columns + ['SalePrice']]


to_encode = housing_columns
 
    
le = preprocessing.LabelEncoder
encoderDict = defaultdict(le)

for column in to_encode:
    df[column] = df[column].dropna()
    df = df[df[column].notnull()]
    df[column] = encoderDict[column].fit_transform(df[column])


df['price_bracket'] = df['SalePrice'].copy().astype(int)
    
classifed_names = []
for bracket in range(0, 15):
    # Each bracket worth 75k
    bracket_width = 100000
    price_min = bracket * bracket_width
#         print(str(bracket) +': '+ str(price_min))
    price_max = price_min + bracket_width
    classifed_names.append(str(price_min) + '-' + str(price_max))
    df.loc[(df['SalePrice'] >= price_min), 'price_bracket'] = bracket

df = df.apply(lambda x: pandas.to_numeric(x.astype(str).str.replace(',',''), errors='coerce'))

df = df.dropna()  
df = df.drop('SalePrice', 1)
# sample = df.sample(n=10000)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('housing.csv')