import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns

def main():

    # Load the dataset
    df = pd.read_csv('datasets/Battery_RUL.csv')

    # Assuming the target variable is 'RUL_label' for classification
    df['RUL_label'] = pd.qcut(df['RUL'], q=[0, 1/3, 2/3, 1], labels=['Label1', 'Label2', 'Label3'])

    # X and y for classification
    X = df.drop(['RUL', 'RUL_label'], axis=1)
    y = df['RUL_label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    # Decision Tree Model without preprocessing
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print('------------ Normal Decision Tree ---------------')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{classification_rep}')

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    # Visualize the Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        filled=True,
        fontsize=10,
        feature_names=X.columns,
        class_names=model.classes_
    )
    plt.title(f'Decision Tree\n\nAccuracy: {accuracy * 100:.2f}%')
    return plt.show()

if __name__ == '__main__':
    main()