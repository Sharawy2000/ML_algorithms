from six import StringIO
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor # Import Decision Tree Regressor
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.metrics import accuracy_score



cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Pre_processing
# Standardize/normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Decision Tree Regression object
clf = DecisionTreeRegressor()

# Train Decision Tree Regression
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))

# Visualization

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,filled=True, rounded=True,special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('decision_tree_output_images/diabetes_Regression.png')
Image(graph.create_png())


