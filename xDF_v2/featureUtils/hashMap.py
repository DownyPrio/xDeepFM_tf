import pandas as pd
from sklearn.datasets import load_boston

import category_encoders as ce

bunch=load_boston()
y=bunch.target
X=pd.DataFrame(bunch.data,columns=bunch.feature_names)
print(X)
print(len(X.columns))
X.info()
print(len(X['RAD'].value_counts()))
enc=ce.HashingEncoder(cols=['RAD']).fit(X)

numeric_dataset=enc.transform(X)

numeric_dataset.info()
print(X.columns)
print(numeric_dataset.columns)
print(X.groupby('RAD').describe())