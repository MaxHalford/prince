import pandas as pd
import matplotlib.pyplot as plt

import os
import prince

org_dir = os.path.dirname(__file__)
absolute_path = os.path.join(org_dir, 'data/decathlon.csv')
# Load a dataframe
df = pd.read_csv(absolute_path, index_col=0)


# Compute the PCA
pca = prince.PCA(df, n_components=-1, supplementary_rows=['Uldal'],
                 supplementary_columns=['Rank', 'Points'])

fig1, ax1 = pca.plot_cumulative_inertia()
fig2, ax2 = pca.plot_rows(color_by='Competition', show_labels=True)
fig3, ax3 = pca.plot_correlation_circle()

plt.show()
