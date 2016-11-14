import pandas as pd
import matplotlib.pyplot as plt

import prince


# Load a dataframe
df = pd.read_csv('doc/data/decathlon.csv', index_col=0)

# Compute the PCA
pca = prince.PCA(df, nbr_components=-1, supplementary_rows=['Uldal'],
                 supplementary_columns=['Rank', 'Points'])

fig1, ax1 = pca.plot_cumulative_inertia()
fig2, ax2 = pca.plot_rows(color_by='Competition', show_labels=True)
fig3, ax3 = pca.plot_correlation_circle()

plt.show()
