import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Load dataset
    df = pd.read_csv('datasets/Battery_RUL.csv')

    # Prepare the data
    X = df.drop('RUL', axis=1)  # Features
    y = df['RUL']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    # Standardize the features
    sc_x = StandardScaler()
    X_train_scaled = sc_x.fit_transform(X_train)
    X_test_scaled = sc_x.transform(X_test)

    # Initialize SVM classifier model
    SVM_classifier = SVR(kernel='rbf')

    # Perform 10-fold cross-validation and evaluate the model
    kf = KFold(n_splits=10, shuffle=True, random_state=None)
    r2_scores = cross_val_score(SVM_classifier, X_train_scaled, y_train, cv=kf, scoring='r2')
    mse_scores = cross_val_score(SVM_classifier, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')

    # Calculate the mean and standard deviation of the scores
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    mean_mse = np.mean(-mse_scores)  # Negate the MSE scores
    std_mse = np.std(-mse_scores)  # Negate the MSE scores

    # Print or log the cross-validated scores
    print(f'Cross-validated R2 scores: {r2_scores}')
    print(f'Mean R2: {mean_r2}, Standard Deviation R2: {std_r2}')
    print(f'Cross-validated MSE scores: {mse_scores}')
    print(f'Mean MSE: {mean_mse}, Standard Deviation MSE: {std_mse}')

    # Fit the model on the training data and predict on the test set
    SVM_classifier.fit(X_train_scaled, y_train)
    y_pred = SVM_classifier.predict(X_test_scaled)

    # Calculate R-squared (R2) and Mean Squared Error (MSE)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Print or log the results
    print(f'R2: {r2}')
    print(f'MSE: {mse}')

    # Visualizing the rbf SVR results
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed

    # Subplot 1: Scatter plot of actual vs. predicted values
    ax1.scatter(y_test, y_pred, color='blue', alpha=0.8, label='Actual vs. Predicted')
    ax1.plot([0, 100], [0, 100], color='red', linewidth=2, linestyle='--', label='Perfect Prediction')
    ax1.set_xlabel('Actual RUL')
    ax1.set_ylabel('Predicted RUL')
    ax1.set_title('RBF SVR Predictions for Battery RUL')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Histogram of prediction errors
    errors = y_test - y_pred
    ax2.hist(errors, bins=20, edgecolor='black')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Errors')

    # Adjust layout for better appearance
    plt.tight_layout()  # Adjust spacing between subplots

    # Show the combined plot
    plt.show()


if __name__ == '__main__':
    main()