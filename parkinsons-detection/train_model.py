import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# Load dataset
data = pd.read_csv("data/parkinsons.data")

# Show first 5 rows
print(data.head())

# Dataset shape
print("Dataset shape:", data.shape)

# Drop 'name' column
data = data.drop(columns=['name'])

X = data.drop(columns=['status'])  # features
y = data['status']                 # target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#svm model
svm_model = SVC(kernel='rbf', C=10, gamma='scale')
svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)

#random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

#logistic regression model 
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)


#model evaluation
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


evaluate_model("SVM", y_test, svm_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("Logistic Regression", y_test, lr_pred)

# Save model and scaler
joblib.dump(svm_model, "model/parkinsons_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully!")


