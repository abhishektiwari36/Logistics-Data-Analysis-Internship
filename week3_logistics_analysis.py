import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Simulation (Hypothetical Logistics Dataset)
np.random.seed(42)
n_shipments = 500

# Generating realistic data
distances = np.random.uniform(50, 2000, n_shipments)
weights = np.random.uniform(10, 500, n_shipments)
modes = np.random.choice(['Road', 'Rail', 'Air'], n_shipments, p=[0.6, 0.3, 0.1])

costs, times = [], []
for i in range(n_shipments):
    dist, mode = distances[i], modes[i]
    if mode == 'Road':
        costs.append(dist * 1.5 + np.random.normal(50, 10))
        times.append(dist / 400 + np.random.normal(1, 0.5))
    elif mode == 'Rail':
        costs.append(dist * 0.8 + np.random.normal(100, 20))
        times.append(dist / 200 + np.random.normal(3, 1))
    else: # Air
        costs.append(dist * 5.0 + np.random.normal(200, 50))
        times.append(dist / 800 + np.random.normal(0.5, 0.2))

# Creating DataFrame
df = pd.DataFrame({
    'Shipment_ID': range(1, n_shipments+1),
    'Distance_km': distances,
    'Weight_kg': weights,
    'Transport_Mode': modes,
    'Cost_USD': np.maximum(costs, 15), # Min cost $15
    'Delivery_Time_Days': np.maximum(times, 0.5) # Min time 0.5 days
})

# 2. Exploratory Data Analysis (EDA)
print("--- Central Tendencies & Distributions ---")
print(df.describe().round(2))
print("\n--- Correlation Matrix ---")
print(df[['Distance_km', 'Weight_kg', 'Cost_USD', 'Delivery_Time_Days']].corr().round(2))

# 3. Visualizations
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# V1: Delivery Time Distribution
sns.histplot(df['Delivery_Time_Days'], bins=25, kde=True, ax=axes[0,0], color='teal')
axes[0,0].set_title('1. Distribution of Delivery Times')
axes[0,0].set_xlabel('Delivery Time (Days)')

# V2: Cost vs Distance by Mode
sns.scatterplot(data=df, x='Distance_km', y='Cost_USD', hue='Transport_Mode', ax=axes[0,1], palette='Set1')
axes[0,1].set_title('2. Cost vs Distance by Transport Mode')

# V3: Average Cost & Time by Mode
sns.barplot(data=df, x='Transport_Mode', y='Cost_USD', ax=axes[1,0], palette='pastel', errorbar=None)
axes[1,0].set_title('3. Average Transportation Cost by Mode')

# V4: Correlation Heatmap
sns.heatmap(df[['Distance_km', 'Weight_kg', 'Cost_USD', 'Delivery_Time_Days']].corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
axes[1,1].set_title('4. Correlation Heatmap')

plt.tight_layout()
plt.show()
