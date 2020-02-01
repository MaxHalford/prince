import os; import sys; sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import prince
from sklearn import datasets


X = pd.read_csv('children.csv', index_col=0)

ca = prince.CA().fit(X)

print('Eigenvalues')
print(ca.eigenvalues_)
print(ca.explained_inertia_)
print('---')

print('U')
print(ca.U_[:5])
print('---')

print('V')
print(ca.V_)
print('---')

print('s')
print(ca.s_)
print('---')

print('Row coords')
print(ca.row_coordinates(X)[:5])
print('---')
