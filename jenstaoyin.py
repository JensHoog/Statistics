import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read the parquet files containing item features and user transactions
df_fea = pd.read_parquet("/content/drive/My Drive/Statisticscode/")
df_trans = pd.read_parquet("/content/drive/My Drive/Statisticscode/")

# Display the first few rows of the item features DataFrame to understand its structure
print(df_fea.head())
