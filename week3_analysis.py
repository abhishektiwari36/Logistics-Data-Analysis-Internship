import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Simulate Logistics Data
data = {
    'Distance_km': [100, 250, 400, 550, 700, 850], 
    'Cost_USD': [150, 300, 450, 650, 900, 1200],
    'Delivery_Hours': [24, 36, 48, 72, 96, 120]
}
df = pd.DataFrame(data)

# 2. Exploratory Data Analysis (EDA)
print(df.describe())

# 3. Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix for Logistics Data')
plt.show()

# 4. Scatter Plot: Distance vs Cost
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Distance_km', y='Cost_USD', data=df, color='blue', s=100)
plt.title('Transportation Cost vs Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Cost (USD)')
plt.grid(True)
plt.show()
