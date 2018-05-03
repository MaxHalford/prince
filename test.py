import pandas as pd
import prince

df = pd.read_csv('output_s5.csv', sep=';', index_col=0)
mca = prince.MCA().fit(df)
print(prince.__version__)
