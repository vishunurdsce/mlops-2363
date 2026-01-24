import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create a noisy classification dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
    n_classes=2, flip_y=0.15, random_state=42
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameter grid
n_estimators_list = [10, 50, 100]
max_depth_list = [3, 5, 7]

# Track best model
best_accuracy = 0
best_model = None
best_params = {}

# Set experiment (optional)
mlflow.set_experiment("noisy_rf_gridsearch")

# Grid search with MLflow tracking
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        with mlflow.start_run():
            # Log hyperparameters
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

            # Track best model
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "accuracy": acc
                }

            # Log the model
            mlflow.sklearn.log_model(model, "rf_model")

# Print best model info
print("\n✅ Best Model Summary")
print(best_params)
