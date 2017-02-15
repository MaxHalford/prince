import pandas as pd

import os
import prince

org_dir = os.path.dirname(__file__)
absolute_path = os.path.join(org_dir, 'data/iris.csv')
df = pd.read_csv(absolute_path)

pca = prince.PCA(df, n_components=4)

fig1, ax1 = pca.plot_cumulative_inertia()
fig2, ax2 = pca.plot_rows(color_by='class', ellipse_fill=True)
fig3, ax3 = pca.plot_correlation_circle()

fig1.savefig('pca_cumulative_inertia.png', bbox_inches='tight', pad_inches=0.5)
fig2.savefig('pca_row_principal_coordinates.png', bbox_inches='tight', pad_inches=0.5)
fig3.savefig('pca_correlation_circle.png', bbox_inches='tight', pad_inches=0.5)
