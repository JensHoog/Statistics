!pip install pyarrow
!pip install fastparquet

from google.colab import drive
drive.mount('/content/drive')

import os
# List contents of the root directory of your Google Drive
root_dir = '/content/drive/My Drive'
for item in os.listdir(root_dir):
    print(item)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read the parquet files containing item features and user transactions
df_fea = pd.read_parquet("/content/drive/My Drive/Statisticscode/TaoYin_Item_features.parquet")
df_trans = pd.read_parquet("/content/drive/My Drive/Statisticscode/TaoYin_User_Transactions.parquet")

# Display the first few rows of the item features DataFrame to understand its structure
print(df_fea.head())

# Replace all NaN values with zero to handle missing data
df_fea = df_fea.fillna(0)

# One-hot encode the specified columns ('statistiek_hoofdgroep', 'statistiek_subgroep') to convert categorical data into binary features
encoded_df = pd.get_dummies(df_fea, columns=['statistiek_hoofdgroep', 'statistiek_subgroep'])

# Convert the one-hot encoded DataFrame to integer type to save memory
encoded_df = encoded_df.astype(int)

# Display the transformed DataFrame after one-hot encoding
print(encoded_df)
