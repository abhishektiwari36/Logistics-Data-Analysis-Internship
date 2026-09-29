import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Simulate loading the dataset
df = pd.read_csv('logistics_transport_data.csv')

# 2. Handling Missing Values (Imputing median for Delivery_Time_Days)
median_delivery = df['Delivery_Time_Days'].median()
df['Delivery_Time_Days'].fillna(median_delivery, inplace=True)

# 3. Outlier Detection & Treatment using IQR for Shipping_Cost
Q1 = df['Shipping_Cost'].quantile(0.25)
Q3 = df['Shipping_Cost'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping outliers to upper and lower bounds
df['Shipping_Cost'] = np.where(df['Shipping_Cost'] > upper_bound, upper_bound,
                      np.where(df['Shipping_Cost'] < lower_bound, lower_bound, df['Shipping_Cost']))

# 4. Normalization of Numerical Features using Min-Max Scaler
scaler = MinMaxScaler()
numerical_cols = ['Shipment_Weight', 'Shipping_Cost']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Displaying the cleaned and preprocessed data
print(df.head())
