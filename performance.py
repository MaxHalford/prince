import cProfile
import pandas as pd
from scipy.sparse import rand

import prince
from sklearn.decomposition import PCA

n = int(1e6)
p = int(1e2)
density = 0.0001
X = pd.DataFrame(rand(n, p, density=density).toarray())

cProfile.run('prince.PCA(X, nbr_components=2)')
cProfile.run("pca = PCA(svd_solver='randomized', n_components=2); pca.fit(X)")
