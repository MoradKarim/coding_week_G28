# Configuration settings for the Medical Decision Support Application

# File paths
DATA_PATH = 'data/bone_marrow_transplant_data.csv'  # Path to the dataset
MODEL_SAVE_PATH = 'models/trained_model.pkl'       # Path to save the trained model
LOG_FILE_PATH = 'logs/app.log'                     # Path to save application logs

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    'xgboost': {
        'learning_rate': 0.05,
        'n_estimators': 100,
        'max_depth': 6,
        'subsample': 0.8
    },
    'lightgbm': {
        'learning_rate': 0.05,
        'n_estimators': 100,
        'max_depth': 6,
        'num_leaves': 31
    }
}

# SHAP configuration (SHAP settings)
SHAP_PARAMS = {
    'max_display': 10  # Max number of features to display in SHAP plots
}

# Preprocessing parameters
PREPROCESSING_PARAMS = {
    'missing_value_strategy': 'mean',  # Options: mean, median, mode
    'outlier_detection_method': 'IQR', # Options: IQR, Z-Score
    'normalize_data': True             # Normalize data before model training
}

# Web application settings (for Streamlit or Flask)
WEB_APP_PARAMS = {
    'title': 'Pediatric Bone Marrow Transplant Success Prediction',
    'input_fields': ['age', 'blood_type', 'previous_treatments', 'wbc_count', 'donor_compatibility', 'health_score'],
    'prediction_threshold': 0.5        # Probability threshold to predict success
}

# Logging configuration
LOGGING = {
    'level': 'INFO',  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Deployment settings (if applicable)
DEPLOYMENT = {
    'platform': 'heroku',  # Options: heroku, aws, gcp
    'api_key': 'your_api_key_here'
}

# Other settings
RANDOM_SEED = 42  # Random seed for reproducibility
