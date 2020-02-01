import os; import sys; sys.path.insert(1, os.path.join(sys.path[0], '..'))

import prince
import pandas as pd


X = pd.read_csv('tea.csv', index_col=0)

mca = prince.MCA(n_components=5).fit(X)

print('Eigenvalues')
print(mca.eigenvalues_)
print(mca.explained_inertia_)
print('---')

print('U')
print(mca.U_[:5])
print('---')

print('V')
print(mca.V_.T[:5])
print('---')

print('s')
print(mca.s_)
print('---')

print('Row coords')
print(mca.row_coordinates(X)[:5])
print('---')
