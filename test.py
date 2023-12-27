import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('datasets/Battery_RUL.csv')

# Prepare the data
X = df.drop('RUL', axis=1)  # Features
y = df['RUL'] # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=20)

# Standardize the features,Preprocessing
#Feature scaling is a preprocessing step in machine learning
#where you transform the features of your dataset to be on a similar scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize KNN regressor model
knn_regressor = KNeighborsRegressor()

# Fit the model
knn_regressor.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_knn_reg = knn_regressor.predict(X_test_scaled)

# Evaluate the model using regression metrics
mse = mean_squared_error(y_test, y_pred_knn_reg)
r2 = r2_score(y_test, y_pred_knn_reg)


# Print regression metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


# Apply cross-validation for regression
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(knn_regressor, X, y, cv=kf, scoring='neg_mean_squared_error')

# Convert the negative MSE to positive
cross_val_scores = -cross_val_scores

# Print cross-validation results
print("\nCross-Validation Scores (MSE):", cross_val_scores)
print("Average MSE (Cross-Validation):", np.mean(cross_val_scores))

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_knn_reg)
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Actual vs Predicted Values')

# Plot residuals
residuals = y_test - y_pred_knn_reg
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')

plt.show()
