import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Separate features (X) and target variable (y)
cancer = load_breast_cancer()
X = cancer.data[:,np.newaxis,0]
y = cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize/normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_score = linear_reg.score(X_test, y_test)

y_pred=linear_reg.predict(X_test)

# Calculate accuracy (for linear regression, accuracy may not be the best metric)
# This is just to show how you might evaluate the model

accuracy = linear_reg.score(X_test, y_test)
print(f"Accuracy: {accuracy}")


# Visualize the linear regression line
plt.scatter(X_test, y_test,alpha=0.6, color='black', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Predictions')
plt.title('Linear Regression')
plt.legend()
plt.show()



