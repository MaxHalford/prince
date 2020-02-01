import os; import sys; sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import prince

X = pd.read_csv('wine.csv', index_col=0)

variables = X.columns.tolist()

group_sizes = {
    'orig': 2,
    'olf': 5,
    'vis': 3,
    'olfag': 10,
    'gust': 9,
    'ens': 2
}

i = 0
groups = {}

for name, n in group_sizes.items():
    groups[name] = variables[i:i + n]
    i += n


mfa = prince.MFA(
    groups=groups,
    n_components=5,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='auto',
    random_state=42
)
mfa = mfa.fit(X)

# print('ORIG')
# print(mfa.partial_factor_analysis_['orig'].eigenvalues_)
# print(mfa.partial_factor_analysis_['orig'].s_)
# print('---')

# print('VIS')
# print(mfa.partial_factor_analysis_['vis'].eigenvalues_)
# print(mfa.partial_factor_analysis_['vis'].s_)
# print('---')

print('Eigenvalues')
print(mfa.eigenvalues_)
print(mfa.explained_inertia_)
print('---')

# print('U')
# print(mfa.U_[:5])
# print('---')

# print('V')
# print(mfa.V_.T[:5])
# print('---')

print('s')
print(mfa.s_)
print('---')

print('Row coords')
print(mfa.row_coordinates(X)[:5])
print('---')
