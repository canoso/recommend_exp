from sklearn.feature_extraction import DictVectorizer
from fastFM import als
import numpy as np

train = [
    {"user": "1", "item": "5", "age": "19"},
    {"user": "2", "item": "43", "age": "33"},
    {"user": "3", "item": "20", "age": "55"},
    {"user": "4", "item": "10", "age": "20"},
]

v = DictVectorizer()
X = v.fit_transform(train)

print(X.toarray())

y = np.array([5.0, 1.0, 2.0, 4.0])
print(y)

fm = als.FMRegression(n_iter=10000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
fm.fit(X, y)

print(fm.predict(v.transform({"user": "5", "item": "10", "age": "24"})))
