import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.catboost

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv(r"C:\Users\AIML\Documents\mlops-2363\data\mlops-theory-main\Unit-1\loan_risk_data.csv")

target_col = "RiskCategory"
X = data.drop(columns=[target_col])
y = data[target_col]

# Identify categorical features
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 2. Start MLflow Experiment
# =========================
mlflow.set_experiment("Loan_Risk_Classification")

with mlflow.start_run():

    # Model parameters
    params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "MultiClass"
    }

    # Log parameters
    mlflow.log_params(params)

    # Train model
    model = CatBoostClassifier(
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        loss_function=params["loss_function"],
        eval_metric="Accuracy",
        verbose=0
    )

    model.fit(X_train, y_train, cat_features=cat_features)

    # Predictions
    y_pred = model.predict(X_test).flatten()

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.catboost.log_model(model, artifact_path="catboost_model")

print("✅ MLflow tracking completed!")
