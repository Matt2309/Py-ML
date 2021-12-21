import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

X = [
    [20, np.nan],
    [np.nan, 'm'],
    [30, 'f'],
    [35, 'f'],
    [np.nan, np.nan]
]

transformers = [
    ['age_imputer', SimpleImputer(), [0]],
    ['sex_imputer', SimpleImputer(strategy='constant', fill_value='n.d'), [1]],
]

ct = ColumnTransformer(transformers)
ct.fit(X)
X = ct.transform(X) # come il predict ma con i trasformatori si usa transform

print(X)