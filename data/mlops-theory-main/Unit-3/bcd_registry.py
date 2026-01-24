import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.tracking import MlflowClient

# Create a noisy classification dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
    n_classes=2, flip_y=0.15, random_state=42
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter grid
n_estimators_list = [10, 50, 100]
max_depth_list = [3, 5, 7]

# Track best
best_accuracy = 0
best_model = None
best_params = {}
best_run_id = None

# Set experiment
mlflow.set_experiment("noisy_rf_gridsearch")

# Start grid search
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Log params
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Train model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Log model
            mlflow.sklearn.log_model(model, "rf_model")

            # Update best model
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "accuracy": acc
                }
                best_run_id = run_id

# Print best model info
print("\n✅ Best Model Summary")
print(best_params)

# Register the best model
if best_run_id:
    model_uri = f"runs:/{best_run_id}/rf_model"
    model_name = "noisy_rf_best_model"

    # Set tag for best run
    client = MlflowClient()
    client.set_tag(best_run_id, "best_model", "true")

    # Register the best model
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # (Optional) Add a description
    client.update_model_version(
        name=result.name,
        version=result.version,
        description=f"Best model with acc={best_accuracy:.4f}, n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}"
    )
