import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    # Load the dataset
    df = pd.read_csv('datasets/Battery_RUL.csv')

    # 1. Data Preprocessing
    # Drop rows with missing values
    df = df.dropna()

    # Drop columns that are not features
    # Assuming 'Cycle_Index' is an identifier and 'RUL' is the target variable
    features = df.drop(['Cycle_Index', 'RUL'], axis=1)
    target = df['RUL']

    # Feature Scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 2. Data Visualization
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # 3. Apply Linear Regression
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=.1, random_state=2000)

    # Initialize the Linear Regression model
    lr_model = LinearRegression()

    # Train the model
    lr_model.fit(X_train, y_train)
    print(lr_model.fit(X_train, y_train))

    # Make predictions
    y_pred = lr_model.predict(X_test)

    # 4. Calculate Accuracy (using R-squared)
    r2 = r2_score(y_test, y_pred)
    print(f'R-squared (Accuracy): {r2}')

    # 5. Calculate R-squared Error (same as accuracy in this context)
    # Note: R-squared is a measure of accuracy in regression, not an error. Higher is better.
    # It is already calculated above, but if you need it as 'R-squared Error', you can use 1 - r2
    r2_error = 1 - r2
    print(f'R-squared Error: {r2_error}')

    # 6. Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot the true vs predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.show()


    # Assuming you have a trained linear regression model named 'lr_model'
    intercept = lr_model.intercept_
    coefficients = lr_model.coef_

    plt.show()

if __name__ == '__main__':
    main()



