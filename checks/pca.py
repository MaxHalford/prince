import os; import sys; sys.path.insert(1, os.path.join(sys.path[0], '..'))

import prince
from sklearn import datasets


X, y = datasets.load_iris(return_X_y=True)

pca = prince.PCA(rescale_with_mean=True, rescale_with_std=True, n_components=2).fit(X)

print('Eigenvalues')
print(pca.eigenvalues_)
print(pca.explained_inertia_)
print('---')

print('U')
print(pca.U_[:5])
print('---')

print('V')
print(pca.V_)
print('---')

print('s')
print(pca.s_)
print('---')

print('Row coords')
print(pca.row_coordinates(X)[:5])
print('---')
