"""
Machine Learning Model Training with Hyperparameter Tuning
Author: Andi Arif Abdillah
Description: Train model with manual MLflow logging and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data():
    """Load preprocessed training and test data"""
    print("Loading preprocessed data...")
    
    train_df = pd.read_csv('iris_train_preprocessed.csv')
    test_df = pd.read_csv('iris_test_preprocessed.csv')
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    print(f"✓ Data loaded - Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Create and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['setosa', 'versicolor', 'virginica'],
                yticklabels=['setosa', 'versicolor', 'virginica'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_feature_importance(model, feature_names, save_path='feature_importance.png'):
    """Create and save feature importance plot"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def train_model_with_tuning():
    """Train model with hyperparameter tuning and manual MLflow logging"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
    # Set experiment
    mlflow.set_experiment("Iris Classification - Andi Arif Abdillah")
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest_Tuning_Manual"):
        
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
        print("=" * 60)
        
        # Initialize base model
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        print("\nPerforming Grid Search...")
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print("\n✓ Grid Search completed")
        print(f"\nBest Parameters: {best_params}")
        print(f"Best CV Score: {best_score:.4f}")
        
        # Log hyperparameters manually
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "accuracy")
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Additional metrics for advanced
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        logloss = log_loss(y_test, y_pred_proba)
        
        # Log metrics manually
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("log_loss", logloss)
        mlflow.log_metric("best_cv_score", best_score)
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1-Score:    {f1:.4f}")
        print(f"ROC AUC:     {roc_auc:.4f}")
        print(f"Log Loss:    {logloss:.4f}")
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=['setosa', 'versicolor', 'virginica'],
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['setosa', 'versicolor', 'virginica']
        ))
        
        # Save classification report as JSON artifact
        report_path = 'classification_report.json'
        with open(report_path, 'w') as f:
            json.dump(class_report, f, indent=4)
        mlflow.log_artifact(report_path)
        
        # Create and log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path)
        
        # Create and log feature importance
        fi_path = plot_feature_importance(best_model, X_train.columns)
        mlflow.log_artifact(fi_path)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name="IrisClassifier_AndiArifAbdillah"
        )
        
        # Log additional artifacts for ADVANCED level
        # 1. Save grid search results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results_path = 'grid_search_results.csv'
        cv_results.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path)
        
        # 2. Save model metadata
        metadata = {
            "model_type": "RandomForestClassifier",
            "author": "Andi Arif Abdillah",
            "dataset": "Iris",
            "n_features": X_train.shape[1],
            "n_classes": len(np.unique(y_train)),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "best_params": best_params,
            "feature_names": list(X_train.columns)
        }
        metadata_path = 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        mlflow.log_artifact(metadata_path)
        
        # Get run info
        run = mlflow.active_run()
        print("\n" + "=" * 60)
        print("MLFLOW RUN INFO")
        print("=" * 60)
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Artifact URI: {run.info.artifact_uri}")
        
        print("\n✓ Model, metrics, and artifacts logged to MLflow")
        print("✓ View results at: http://127.0.0.1:5000")
        
        # Return model and metrics for verification
        return best_model, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'log_loss': logloss
        }


if __name__ == "__main__":
    model, metrics = train_model_with_tuning()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)