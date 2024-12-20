# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load the dataset
file_path ='C:/Users/Rupesh Shinde/Desktop/CRM/sweets_segmentation_data.csv'  # Updated file path to match uploaded file
df = pd.read_csv(file_path)

# Display basic info and first few rows of the dataset
print("Data Overview:")
print(df.info())
print("\nSample data:")
print(df.head())

# Handle missing values (if any)
df = df.dropna()
print(f"\nData shape after dropping missing values: {df.shape}")

# Encode categorical variables using Label Encoding
categorical_columns = ['Region', 'Sweet_Preference', 'Festival', 'Language', 'Socio_Economic_Class']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Standardize the 'Purchase_Amount' column
scaler = StandardScaler()
df['Purchase_Amount'] = scaler.fit_transform(df[['Purchase_Amount']])

# Replace infinite values with NaN and drop or handle them
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Confirm preprocessing
print("\nPreprocessing completed. Data shape:", df.shape)
print("Sample data after preprocessing:")
print(df.head())

# Exploratory Data Analysis (EDA)
# Plot the distribution of purchase amounts
plt.figure(figsize=(8, 5))
sns.histplot(df['Purchase_Amount'], kde=True, bins=30)
plt.title('Distribution of Purchase Amounts')
plt.xlabel('Standardized Purchase Amount')
plt.ylabel('Frequency')
plt.show()
print("Distribution of Purchase Amounts plotted.")

# Plot the count of sweet preferences
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Sweet_Preference', order=df['Sweet_Preference'].value_counts().index)
plt.title('Count of Sweet Preferences')
plt.xticks(rotation=45)
plt.xlabel('Sweet Preference')
plt.ylabel('Count')
plt.show()
print("Count of Sweet Preferences plotted.")

# Clustering Model Development
# Select features for clustering
features = df[categorical_columns + ['Purchase_Amount']]

# Verify features shape
print("Feature data shape:", features.shape)

# Determine the optimal number of clusters using the Elbow Method and Silhouette Score
inertia = []
silhouette_scores = []
k_range = range(2, 11)

print("Calculating inertia and silhouette scores for different cluster counts...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features, kmeans.labels_))
    print(f"Processed {k} clusters: Inertia = {kmeans.inertia_}, Silhouette Score = {silhouette_scores[-1]:.4f}")

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method - Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid()
plt.show()
print("Elbow Method plot displayed.")

# Plot the Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Scores vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid()
plt.show()
print("Silhouette Scores plot displayed.")

# Optimal number of clusters (based on the highest silhouette score)
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters: {optimal_k}")

# Train the final model with the optimal number of clusters
try:
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_kmeans.fit(features)

    # Add the cluster labels to the dataframe
    df['Cluster_Label'] = final_kmeans.labels_

    # Calculate and print the final silhouette score
    final_silhouette_score = silhouette_score(features, final_kmeans.labels_)
    print(f"Final Silhouette Score: {final_silhouette_score:.4f}")

    # Display the number of customers in each cluster
    cluster_counts = df['Cluster_Label'].value_counts()
    print("Number of customers in each cluster:")
    print(cluster_counts)

except Exception as e:
    print(f"Error in training the final model: {e}")

# Visualize Clusters using PCA
try:
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)

    # Plot the clusters
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=df['Cluster_Label'], cmap='viridis', alpha=0.6)
    plt.title('Customer Segments Visualized using PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.show()
    print("Clusters visualized using PCA successfully.")

except Exception as e:
    print(f"Error in PCA visualization: {e}")

# Decode categorical columns back to original text values
for col, le in label_encoders.items():
    df[col] = le.inverse_transform(df[col])

# Analyze key characteristics of each cluster (e.g., average purchase amount)
try:
    # Select only numeric columns for the summary
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    cluster_summary = df.groupby('Cluster_Label')[numeric_columns].mean()
    
    print("Cluster Summary:")
    print(cluster_summary)

    # Save the labeled dataset to a new CSV with decoded labels
    output_file = r"C:\Users\Rupesh Shinde\Desktop\CRM\sweets_segmented_data.csv"
    df.to_csv(output_file, index=False)
    print("Segmented data saved successfully with original labels.")

except Exception as e:
    print(f"Error in analyzing clusters or saving results: {e}")
