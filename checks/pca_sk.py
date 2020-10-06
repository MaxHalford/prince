import os; import sys; sys.path.insert(1, os.path.join(sys.path[0], '..'))

from sklearn import datasets
from sklearn import decomposition
from sklearn import preprocessing


X, y = datasets.load_iris(return_X_y=True)
X = preprocessing.scale(X)

pca = decomposition.PCA(n_components=4)
pca.fit(X)

print('Eigenvalues')
print(pca.explained_variance_ * (149) / 150)
print('---')

print('V')
print(pca.components_)
print('---')

print('s')
print(pca.singular_values_)
print('---')

print('Row coords')
print(pca.transform(X)[:5])
print('---')
