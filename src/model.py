import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv(r"C:\Users\AIML\Documents\mlops-2363\data\mlops-theory-main\Unit-1\loan_risk_data.csv")

print("Data Shape:", data.shape)
print("Columns:", data.columns)

# =========================
# 2. Features & Target
# =========================
target_col = "RiskCategory"
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
# 5. Train Model (Multi-class)
# =========================
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    verbose=100
)

model.fit(X_train, y_train, cat_features=cat_features)

# =========================
# 6. Evaluate Model
# =========================
y_pred = model.predict(X_test).flatten()

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 7. Predict on FULL DATASET (optional)
# =========================
all_preds = model.predict(X).flatten()

data["Predicted_Risk"] = all_preds

# Save predictions
data.to_csv("loan_risk_with_predictions.csv", index=False)
print("\nPredictions saved to loan_risk_with_predictions.csv ✅")
