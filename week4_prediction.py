import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Data Simulation
np.random.seed(42)
df = pd.DataFrame({
    'Distance_km': np.random.randint(50, 1000, 100),
    'Traffic_Index': np.random.randint(1, 10, 100),
    'Weather_Severity': np.random.randint(1, 5, 100)
})
df['Delivery_Time_Hours'] = (df['Distance_km'] / 50) + (df['Traffic_Index'] * 2) + (df['Weather_Severity'] * 3) + np.random.normal(0, 2, 100)

# 2. Data Preparation
X = df[['Distance_km', 'Traffic_Index', 'Weather_Severity']]
y = df['Delivery_Time_Hours']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training (Ensemble Method)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Testing and Evaluation
predictions = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print(f"Model Performance -> RMSE: {rmse:.2f}, R-squared: {r2:.2f}")

# 5. Cross Validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"5-Fold Cross-Validation R2 Mean: {cv_scores.mean():.2f}")
