# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()

#Encode the categorical column
label_encoder = LabelEncoder()
penguins_df['sex'] = label_encoder.fit_transform(penguins_df['sex'])# chance this later
#penguins_df.head()

#Scale the numerical data 
scaler = StandardScaler()
df_scaled = scaler.fit_transform(penguins_df)

#Apply Kmeans

kmeans = KMeans(n_clusters=3, random_state=42)
penguins_df['Cluster'] = kmeans.fit_predict(df_scaled)
stat_penguins = penguins_df.groupby('Cluster').mean()
print(stat_penguins)