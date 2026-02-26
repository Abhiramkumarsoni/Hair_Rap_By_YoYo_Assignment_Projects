"""
Model Training Script for No-Show Prediction Engine.
Trains, compares, and saves the best ML model.
Integrates with MLflow for experiment tracking and model registry.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import tempfile
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import prepare_features, get_feature_names

# ─── MLflow Configuration ────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.path.join(PROJECT_ROOT, "mlruns")
EXPERIMENT_NAME = "NoShow_Prediction"


def train_and_evaluate():
    """Train multiple models, compare them, and save the best one with MLflow tracking."""

    print("=" * 60)
    print("  No-Show Prediction — Model Training Pipeline")
    print("  📊 MLflow Experiment Tracking Enabled")
    print("=" * 60)

    # ─── Setup MLflow ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n  MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"  MLflow experiment:   {EXPERIMENT_NAME}")

    # ─── Load Data ────────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    data_path = os.path.join(PROJECT_ROOT, "data", "salon_bookings.csv")
    df = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(df):,} records")

    # ─── Feature Engineering ──────────────────────────────────────────────
    print("\n[2/6] Engineering features...")
    X, y, label_encoders = prepare_features(df, fit=True)
    print(f"  ✓ Features: {X.shape[1]} columns")
    print(f"  ✓ Target distribution: Show={int((y == 0).sum()):,} | No-Show={int(y.sum()):,}")

    # ─── Train/Test Split ─────────────────────────────────────────────────
    print("\n[3/6] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ─── Define Models ────────────────────────────────────────────────────
    print("\n[4/6] Training models with MLflow tracking...")

    # Calculate scale_pos_weight for imbalanced classes
    n_show = int((y == 0).sum())
    n_noshow = int(y.sum())
    scale_weight = n_show / n_noshow  # ~3.2x

    models = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "params": {"max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs"},
        },
        "Random_Forest": {
            "model": RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                random_state=42, n_jobs=-1, class_weight="balanced"
            ),
            "params": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced"},
        },
    }

    # Try to import XGBoost and LightGBM
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = {
            "model": XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_weight, subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric="logloss", verbosity=0
            ),
            "params": {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "scale_pos_weight": round(scale_weight, 2), "subsample": 0.8, "colsample_bytree": 0.8,
            },
        }
    except ImportError:
        print("  ⚠ XGBoost not available, skipping")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = {
            "model": LGBMClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                is_unbalance=True, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            ),
            "params": {
                "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
                "is_unbalance": True, "subsample": 0.8, "colsample_bytree": 0.8,
            },
        }
    except ImportError:
        print("  ⚠ LightGBM not available, skipping")

    results = {}
    trained_models = {}
    run_ids = {}

    # ─── Train Each Model with MLflow Tracking ───────────────────────────
    for name, config in models.items():
        model_obj = config["model"]
        params = config["params"]

        print(f"\n  Training {name}...")

        with mlflow.start_run(run_name=name) as run:
            # Log parameters
            mlflow.log_param("model_type", name)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("no_show_rate", round(y.mean(), 4))
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Train
            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)
            y_proba = model_obj.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "auc_roc": roc_auc_score(y_test, y_proba),
            }

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model to MLflow
            signature = infer_signature(X_train, model_obj.predict(X_train))
            mlflow.sklearn.log_model(
                model_obj,
                name="model",
                signature=signature,
                input_example=X_train.head(1),
            )

            # Log feature importances as artifact
            feature_name_map = get_feature_names()
            if hasattr(model_obj, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature": X.columns,
                    "feature_display": [feature_name_map.get(f, f) for f in X.columns],
                    "importance": model_obj.feature_importances_
                }).sort_values("importance", ascending=False)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, prefix=f'fi_{name}_') as tmp:
                    fi.to_csv(tmp.name, index=False)
                    mlflow.log_artifact(tmp.name, artifact_path="feature_importance")
            elif hasattr(model_obj, "coef_"):
                fi = pd.DataFrame({
                    "feature": X.columns,
                    "feature_display": [feature_name_map.get(f, f) for f in X.columns],
                    "importance": np.abs(model_obj.coef_[0])
                }).sort_values("importance", ascending=False)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, prefix=f'fi_{name}_') as tmp:
                    fi.to_csv(tmp.name, index=False)
                    mlflow.log_artifact(tmp.name, artifact_path="feature_importance")

            # Log confusion matrix as artifact
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual Show", "Actual No-Show"], columns=["Pred Show", "Pred No-Show"])
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, prefix=f'cm_{name}_') as tmp:
                cm_df.to_csv(tmp.name)
                mlflow.log_artifact(tmp.name, artifact_path="confusion_matrix")

            # Store results
            results[name] = {
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "AUC-ROC": metrics["auc_roc"],
            }
            trained_models[name] = model_obj
            run_ids[name] = run.info.run_id

            print(f"    AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics['f1_score']:.4f} | Acc: {metrics['accuracy']:.4f}")
            print(f"    📝 MLflow Run ID: {run.info.run_id}")

    # ─── Model Comparison ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("AUC-ROC", ascending=False)
    print(results_df.to_string())

    # ─── Select Best Model via MLflow ─────────────────────────────────────
    print("\n  🔍 Querying MLflow for best run (highest AUC-ROC)...")
    best_run_df = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="",
        order_by=["metrics.auc_roc DESC"],
        max_results=1,
    )
    best_run_id = best_run_df.iloc[0]["run_id"]
    best_name = best_run_df.iloc[0]["tags.mlflow.runName"]
    best_model = trained_models[best_name]
    best_metrics = results[best_name]

    print(f"\n  🏆 Best Model (via MLflow): {best_name} (AUC-ROC: {best_metrics['AUC-ROC']:.4f})")
    print(f"  📝 Best Run ID: {best_run_id}")

    # Register best model in MLflow Model Registry & set stage to Production
    print("\n[5/6] Registering best model in MLflow Model Registry...")
    model_uri = f"runs:/{best_run_id}/model"
    try:
        registered_model = mlflow.register_model(model_uri, "noshow_predictor")
        print(f"  ✓ Registered as: noshow_predictor (version {registered_model.version})")

        # Promote to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="noshow_predictor",
            version=registered_model.version,
            stage="Production",
        )
        print(f"  ✓ Stage set to: Production")
    except Exception as e:
        print(f"  ⚠ Model registry note: {e}")
        print(f"  ✓ Model still tracked via run ID: {best_run_id}")

    # ─── Save Artifacts (also keep local copies for dashboard/API) ────────
    print("\n[6/6] Saving local model artifacts...")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save best model
    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"  ✓ Model saved: {model_path}")

    # Save label encoders
    encoders_path = os.path.join(models_dir, "label_encoders.pkl")
    joblib.dump(label_encoders, encoders_path)
    print(f"  ✓ Encoders saved: {encoders_path}")

    # Save comparison results
    results_path = os.path.join(models_dir, "model_comparison.csv")
    results_df.to_csv(results_path)
    print(f"  ✓ Comparison saved: {results_path}")

    # Save feature importances (from best model)
    feature_importance_path = os.path.join(models_dir, "feature_importance.csv")
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": X.columns,
            "feature_display": [feature_name_map.get(f, f) for f in X.columns],
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(feature_importance_path, index=False)
        print(f"  ✓ Feature importances saved: {feature_importance_path}")

        print("\n  Top 10 Features:")
        for _, row in fi.head(10).iterrows():
            bar = "█" * int(row["importance"] * 100)
            print(f"    {row['feature_display']:35s} {row['importance']:.4f} {bar}")
    elif hasattr(best_model, "coef_"):
        fi = pd.DataFrame({
            "feature": X.columns,
            "feature_display": [feature_name_map.get(f, f) for f in X.columns],
            "importance": np.abs(best_model.coef_[0])
        }).sort_values("importance", ascending=False)
        fi.to_csv(feature_importance_path, index=False)
        print(f"  ✓ Feature importances (coefficients) saved: {feature_importance_path}")

    # Save metadata
    metadata = {
        "best_model_name": best_name,
        "metrics": best_metrics,
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": list(X.columns),
        "mlflow_run_id": best_run_id,
        "mlflow_experiment": EXPERIMENT_NAME,
    }
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: {metadata_path}")

    # Save confusion matrix for best model
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    cm_path = os.path.join(models_dir, "confusion_matrix.csv")
    pd.DataFrame(cm, index=["Actual Show", "Actual No-Show"], columns=["Pred Show", "Pred No-Show"]).to_csv(cm_path)
    print(f"  ✓ Confusion matrix saved: {cm_path}")

    print("\n" + "=" * 60)
    print("  ✅ Training pipeline complete!")
    print(f"  📊 View MLflow UI: mlflow ui --backend-store-uri file:///{MLFLOW_TRACKING_URI}")
    print("=" * 60)

    return best_model, results_df


if __name__ == "__main__":
    train_and_evaluate()
