import matplotlib.pyplot as plt
import pandas as pd

import prince

wines = {
    1: 'barolo',
    2: 'grignolino',
    3: 'barbera'
}

df = pd.read_csv('doc/data/wine.csv')
df['kind'] = df['class'].apply(lambda x: wines[x])
del df['class']

pca = prince.PCA(df, nbr_components=-1)

fig1, ax1 = pca.plot_cumulative_inertia(threshold=0.8)
fig2, ax2 = pca.plot_rows(show_points=True, show_labels=False, color_by='kind', ellipse_fill=True)
fig3, ax3 = pca.plot_correlation_circle(axes=(0, 1))

plt.show()
