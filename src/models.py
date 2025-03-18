import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from src.utils.data_processing import optimize_memory

# Function to train Random Forest Classifier
def train_random_forest(X_train, y_train, X_test, y_test, params=None):
    rf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    rf.fit(X_train, y_train)

    # Model evaluation
    y_pred = rf.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print performance metrics
    print(f"Random Forest Performance:")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    return rf

# Function to train XGBoost Classifier
def train_xgboost(X_train, y_train, X_test, y_test, params=None):
    xgb = XGBClassifier(learning_rate=params['learning_rate'], 
                        n_estimators=params['n_estimators'], 
                        max_depth=params['max_depth'])
    xgb.fit(X_train, y_train)

    # Model evaluation
    y_pred = xgb.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print performance metrics
    print(f"XGBoost Performance:")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    return xgb

# Function to train LightGBM Classifier
def train_lightgbm(X_train, y_train, X_test, y_test, params=None):
    lgbm = lgb.LGBMClassifier(learning_rate=params['learning_rate'],
                              n_estimators=params['n_estimators'],
                              max_depth=params['max_depth'],
                              num_leaves=params['num_leaves'])
    lgbm.fit(X_train, y_train)

    # Model evaluation
    y_pred = lgbm.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print performance metrics
    print(f"LightGBM Performance:")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    return lgbm

# Function to train Support Vector Machine (SVM)
def train_svm(X_train, y_train, X_test, y_test, params=None):
    svm = SVC(kernel=params['kernel'], C=params['C'], probability=True)
    svm.fit(X_train, y_train)

    # Model evaluation
    y_pred = svm.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print performance metrics
    print(f"SVM Performance:")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    return svm

# Function to save the trained model
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Function to load a saved model
def load_model(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

# Main function to train, evaluate and select the best model
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Hyperparameters for different models
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10
    }

    xgb_params = {
        'learning_rate': 0.05,
        'n_estimators': 100,
        'max_depth': 6
    }

    lgbm_params = {
        'learning_rate': 0.05,
        'n_estimators': 100,
        'max_depth': 6,
        'num_leaves': 31
    }

    svm_params = {
        'kernel': 'rbf',
        'C': 1
    }

    # Train the models
    rf_model = train_random_forest(X_train, y_train, X_test, y_test, rf_params)
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test, xgb_params)
    lgbm_model = train_lightgbm(X_train, y_train, X_test, y_test, lgbm_params)
    svm_model = train_svm(X_train, y_train, X_test, y_test, svm_params)

    # Save the best model (here, assuming LightGBM performed the best based on evaluation)
    save_model(lgbm_model, 'models/lgbm_model.pkl')

    return lgbm_model  # Return the best model (LightGBM)

# Example: Splitting the dataset, training and selecting the best model
if __name__ == "__main__":
    # Load your dataset and preprocess
    df = pd.read_csv('data/bone_marrow_transplant_data.csv')  # Replace with the actual path
    df = optimize_memory(df)  # Optimize memory usage

    # Assume the target column is 'target_column' (replace it with the actual column name)
    X = df.drop('target_column', axis=1)
    y = df['target_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models and evaluate
    best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)
