from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


cancer = load_breast_cancer()

# Handle missing values if any
# data = cancer.dropna()  # You may choose a different strategy based on your data

# Separate features (X) and target variable (y)
X = cancer.data[:, :2]
y = cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pre_processing
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
