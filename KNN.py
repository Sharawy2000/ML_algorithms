import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score ,auc,roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

def get_percent_res(result):
    res=round((result*100),2)
    res=str(res)+"%"
    return res

# labels=['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)',
#        'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)',
#        'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)',
#        'RUL']
# Load dataset
df = pd.read_csv('datasets/Battery_RUL.csv')
print(df.columns)
# Prepare the data

X = df.drop('RUL', axis=1)  # Features
y = df['RUL']               # Target variable


# Convert RUL to classes (you can adjust the bins or labels as needed)

# For simplicity, let's use two classes: RUL > 50 and RUL <= 50
y_class = (y > 50).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=64)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Apply K-Nearest Neighbors for Classification
# Initialize KNN classifier model
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn_classifier.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_knn_class = knn_classifier.predict(X_test_scaled)

# Evaluate the model using accuracy
accuracy_knn_class = accuracy_score(y_test, y_pred_knn_class)

# Assuming you have the true labels (y_test) and predicted labels (y_pred_knn_class)
# Replace these variables with your actual test labels and predicted labels
plt.subplot(1,2,1)
cm = confusion_matrix(y_test, y_pred_knn_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Classification Report
print('Classification Report:\n', classification_report(y_test, y_pred_knn_class))

# True positive, true negative, false positive, false negative
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn_class).ravel()

# Sensitivity (Recall)
sensitivity = tp / (tp + fn)

# Specificity
specificity = tn / (tn + fp)

# Precision
precision = tp / (tp + fp)

# Recall (Sensitivity)
recall = tp / (tp + fn)

# Print or log the results
print(f'Accuracy (KNN Classification): {get_percent_res(accuracy_knn_class)}')
print(f'Sensitivity : {get_percent_res(sensitivity)}')
print(f'Specificity: {get_percent_res(specificity)}')
print(f'Precision: {get_percent_res(precision)}')
print(f'Recall : {get_percent_res(recall)}')


# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn_class)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.subplot(1,2,2)
# plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(knn_classifier, X, y, cv=kf, scoring='accuracy')

# Print cross-validation results
print("\nCross-Validation Scores:", cross_val_scores)
print("Average Accuracy (Cross-Validation):", np.mean(cross_val_scores))
