import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():

    # Load the dataset
    df = pd.read_csv('datasets/Battery_RUL.csv')

    # Assuming the target variable is 'RUL' (Remaining Useful Life)
    X = df.drop(['RUL'], axis=1)
    y = df['RUL']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Decision Tree Model with preprocessing
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', DecisionTreeRegressor(random_state=42))])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)  # MAE
    r2 = r2_score(y_test, y_pred)
    print('------------ Decision Tree with Preprocessing ---------------')
    print(f'Accuracy Percentage: {r2 * 100:.2f}%')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Test Set Mean Absolute Error (MAE): {mae}')

    # Pruned model with preprocessing
    pruned_model = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', DecisionTreeRegressor(max_depth=4, min_samples_split=5, random_state=42))])

    pruned_model.fit(X_train, y_train)
    y_pred_pruned = pruned_model.predict(X_test)

    # Evaluate the pruned model
    mse_pruned = mean_squared_error(y_test, y_pred_pruned)
    mae_pruned = mean_absolute_error(y_test, y_pred_pruned)
    r2_pruned = r2_score(y_test, y_pred_pruned)

    print('------------ Pruned Model with Preprocessing ---------------')
    print(f'Accuracy Percentage: {r2_pruned * 100:.2f}%')
    print(f'Mean Squared Error (MSE): {mse_pruned}')
    print(f'Mean Absolute Error (MAE): {mae_pruned}')

    # Visualize the pruned Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        pruned_model.named_steps['regressor'],
        filled=True,
        fontsize=10
    )
    plt.title(f'Pruned Decision Tree with Preprocessing\n\nMSE: {mse_pruned:.2f}, MAE: {mae_pruned:.2f},\nAccuracy: {r2_pruned * 100:.2f}%')
    plt.show()

    # Residual plots
    plt.scatter(y_test, y_test - y_pred, color='black', alpha=0.5)
    plt.title(f"Decision Tree with Preprocessing")
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    plt.show()

    plt.scatter(y_test, y_test - y_pred_pruned, color='green', alpha=0.5)
    plt.title(f"Pruned Decision Tree with Preprocessing")
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='blue', linestyle='--', linewidth=2)
    plt.show()

if __name__ == '__main__':
    main()


