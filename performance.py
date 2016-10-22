import cProfile
import pandas as pd
from scipy.sparse import rand

import prince


n = int(1e6)
p = int(1e2)
density = 0.0001
data = pd.DataFrame(rand(n, p, density=density).toarray())

cProfile.run('prince.PCA(data, nbr_components=2)')
