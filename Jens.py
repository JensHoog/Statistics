import pandas as pd

# Set the correct path to your Parquet file
parquet_file_path = '/content/drive/My Drive/Statisticscode/TaoYin_Item_features.parquet'

# Load the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file_path)

# Display the first few rows of the DataFrame
df.head()
