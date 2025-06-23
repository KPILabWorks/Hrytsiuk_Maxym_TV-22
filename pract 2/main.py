import pandas as pd
import dask.dataframe as dd
import time

file_path = 'data/customers.csv'

# ==== Pandas ====
start = time.time()
df_pandas = pd.read_csv(file_path)
end = time.time()
print("Pandas:")
print(df_pandas.head())
print(f"Час завантаження: {end - start:.4f} сек.\n")

# ==== Dask ====
start = time.time()
df_dask = dd.read_csv(file_path)
df_dask_result = df_dask.head()  
end = time.time()
print("Dask:")
print(df_dask_result)
print(f"Час завантаження: {end - start:.4f} сек.")

