import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Load Training Data
# =========================
train_path = r"C:\Users\AIML\Documents\mlops-2363\data\mlops-theory-main\Unit-1\loan_risk_data.csv"
data = pd.read_csv(train_path)

print("Training Data Shape:", data.shape)
print("Columns:", data.columns)

# =========================
# 2. Separate Features & Target
# =========================
target_col = "RiskCategory"   # change if your target column name is different
X = data.drop(columns=[target_col])
y = data[target_col]

# =========================
# 3. Identify Categorical Features
# =========================
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

print("Categorical Feature Indexes:", cat_features)

# =========================
# 4. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. Train CatBoost Model
# =========================
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',   # binary classification
    eval_metric='AUC',
    verbose=100
)

model.fit(X_train, y_train, cat_features=cat_features)

# =========================
# 6. Evaluate Model
# =========================
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 7. Load Batch Prediction Data
# =========================
batch_path = r"C:\Users\AIML\Documents\mlops-2363\data\mlops-theory-main\Unit-1\batch_predictions.csv"
batch_data = pd.read_csv(batch_path)

print("Batch Data Shape:", batch_data.shape)

# =========================
# 8. Predict on Batch Data
# =========================
batch_preds = model.predict(batch_data).flatten()


# =========================
# 9. Save Predictions
# =========================
output = batch_data.copy()
output["Predicted_Risk"] = batch_preds

output.to_csv("batch_predictions_output.csv", index=False)
print("\nPredictions saved to batch_predictions_output.csv ✅")
