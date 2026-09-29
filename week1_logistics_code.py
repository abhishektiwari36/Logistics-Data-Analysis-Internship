import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load the logistics data
data = pd.read_csv('logistics_deliveries.csv')

# 2. Extract geographical coordinates for clustering
coordinates = data[['latitude', 'longitude']]

# 3. Apply K-Means Clustering (assuming we have 5 delivery agents available)
num_agents = 5
kmeans = KMeans(n_clusters=num_agents, random_state=42)
data['delivery_zone'] = kmeans.fit_predict(coordinates)

# 4. View assigned zones for the orders
print(data[['order_id', 'delivery_zone']].head())

# 5. Visualize the Delivery Zones
plt.scatter(data['longitude'], data['latitude'], c=data['delivery_zone'], cmap='viridis')
plt.title('Optimized Delivery Zones for Last-Mile Routing')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
