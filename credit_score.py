import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# 1. Generate a synthetic financial dataset (Income, Debts, Payment History)
print("Loading financial dataset...")
X, y = make_classification(
    n_samples=1000, 
    n_features=4, 
    n_informative=3, 
    n_redundant=1,
    random_state=42, 
    weights=[0.3, 0.7] # Simulates 30% Bad Credit, 70% Good Credit
)

# Map features to the task's required data points
feature_names = ['Annual_Income', 'Current_Debt', 'Late_Payments', 'Credit_Utilization']
data = pd.DataFrame(X, columns=feature_names)
data['Creditworthiness'] = y # 1: Good Credit, 0: Bad Credit

X_features = data.drop('Creditworthiness', axis=1)
y_target = data['Creditworthiness']

# 2. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# 3. Train the Logistic Regression Model
print("Training Logistic Regression Model...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Extracts probabilities required for ROC-AUC

# 5. Evaluate the model using Precision, Recall, F1-Score, and ROC-AUC
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Bad Credit (0)', 'Good Credit (1)']))