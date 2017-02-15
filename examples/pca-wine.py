import matplotlib.pyplot as plt
import pandas as pd

import os
import prince

wines = {
    1: 'barolo',
    2: 'grignolino',
    3: 'barbera'
}

org_dir = os.path.dirname(__file__)
absolute_path = os.path.join(org_dir, 'data/wine.csv')
df = pd.read_csv(absolute_path)

df['kind'] = df['class'].apply(lambda x: wines[x])
df.drop('class', axis=1, inplace=True)

pca = prince.PCA(df, n_components=-1)

fig1, ax1 = pca.plot_cumulative_inertia(threshold=0.8)
fig2, ax2 = pca.plot_rows(show_points=True, show_labels=False, color_by='kind', ellipse_fill=True)
fig3, ax3 = pca.plot_correlation_circle(axes=(0, 1))

plt.show()
