import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read the parquet files containing item features and user transactions
df_fea = pd.read_parquet("C:/Master/Statistics/Sub Assignment 1 attached files 15 October 2024 1018/Tao Yin_Item_features.parquet")
df_trans = pd.read_parquet("C:/Master/Statistics/Sub Assignment 1 attached files 15 October 2024 1018/TaoYin_User_Transactions.parquet")

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

# Select all columns except the first one and the last 56 columns for further processing
selected_data = encoded_df.iloc[:, 1:-56]

# Standardize the selected data to have mean=0 and variance=1, which is important for PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

# Instantiate PCA to keep enough components to explain 95% of the variance in the data
pca = PCA(n_components=0.95)

# Fit and transform the data using PCA to reduce dimensionality
principal_components = pca.fit_transform(scaled_data)

# Convert the principal components to a DataFrame for easier handling and analysis, adding proper column names
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# Fit PCA without reducing components first to visualize explained variance and determine the optimal number of components
pca_full = PCA()
pca_full.fit(scaled_data)

# Calculate the explained variance ratio for each principal component
cumulative_explained_variance = pca_full.explained_variance_ratio_

# Plot the explained variance to create a scree plot for visualization
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot - Explained Variance vs Number of Components')
plt.grid()
plt.show()

# Adjust PCA to keep a reasonable number of components based on the scree plot analysis
pca = PCA(n_components=20)  # This value is based on where the explained variance starts to level off in the scree plot

# Fit and transform the data again using the adjusted number of components
principal_components = pca.fit_transform(scaled_data)

# Convert the principal components to a DataFrame for easier handling and analysis, adding proper column names
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# Add back the columns that were not used in PCA (first column and the last 56 columns)
columns_to_add_back = df_fea.iloc[:, [0] + list(range(-56, 0))]

# Concatenate the retained columns with the PCA-reduced DataFrame to create the final DataFrame
final_df = pd.concat([columns_to_add_back.reset_index(drop=True), pca_df], axis=1)

# Display the first few rows of the final DataFrame to verify the result
print(final_df.head())
