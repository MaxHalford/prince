import matplotlib.pyplot as plt
import pandas as pd

import os
import prince

org_dir = os.path.dirname(__file__)
absolute_path = os.path.join(org_dir, 'data/presidentielles07.csv')
df = pd.read_csv(absolute_path, index_col=0)

ca = prince.CA(df, n_components=-1)

fig1, ax1 = ca.plot_cumulative_inertia()
fig2, ax2 = ca.plot_rows_columns(show_row_labels=True, show_column_labels=True)

plt.show()
