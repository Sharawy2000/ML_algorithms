import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC

# Load your dataset (replace 'your_dataset.csv' with the actual file name)
# data = load_iris()

# Handle missing values if any
# data = data.dropna()  # You may choose a different strategy based on your data

# Separate features (X) and target variable (y)
cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize/normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Nearest Neighbors (KNN)
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_accuracy = knn_classifier.score(X_test, y_test)

plt.subplot(3,1,1)
plt.scatter(X_train[y_train == 0, 0],
            X_train[y_train == 0, 1],
            marker='o',
            label='class 0 (Setosa)')

plt.scatter(X_train[y_train == 1, 0],
            X_train[y_train == 1, 1],
            marker='^',
            label='class 1 (Versicolor)')

plt.scatter(X_train[y_train == 2, 0],
            X_train[y_train == 2, 1],
            marker='s',
            label='class 2 (Virginica)')

plt.legend(loc='upper left')


plt.subplot(3,1,2)
plot_decision_regions(X_train, y_train, knn_classifier)
plt.legend(loc='upper left')
# plt.show()

plt.subplot(3,1,3)

plot_decision_regions(X_test, y_test, knn_classifier)
plt.legend(loc='upper left')

plt.show()


# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_score = linear_reg.score(X_test, y_test)

# Decision Tree Classification
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_classifier_accuracy = dt_classifier.score(X_test, y_test)

# Decision Tree Regression
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)
dt_regressor_score = dt_regressor.score(X_test, y_test)

# Support Vector Machine (SVM)
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svm_accuracy = svm_classifier.score(X_test, y_test)

# You can access the accuracy/score of each model and further evaluate their performance
print(f'KNN Accuracy: {knn_accuracy}')
print(f'Linear Regression Score: {linear_reg_score}')
print(f'Decision Tree Classification Accuracy: {dt_classifier_accuracy}')
print(f'Decision Tree Regression Score: {dt_regressor_score}')
print(f'SVM Accuracy: {svm_accuracy}')