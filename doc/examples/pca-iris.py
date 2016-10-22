import pandas as pd

import prince


df = pd.read_csv('examples/data/iris.csv')

pca = prince.PCA(df, nbr_components=4)

fig1, ax1 = pca.plot_cumulative_inertia()
fig2, ax2 = pca.plot_rows(color_by='class', ellipse_fill=True)

fig1.savefig('cumulative_inertia.png', bbox_inches='tight', pad_inches=0.5)
fig2.savefig('row_projections.png', bbox_inches='tight', pad_inches=0.5)
