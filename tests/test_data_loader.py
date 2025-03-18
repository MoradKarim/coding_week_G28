import pytest
import pandas as pd
from src.data_loader import load_data, preprocess_data
from src.utils.data_processing import optimize_memory

# Test loading the dataset
def test_load_data():
    data = load_data()  # Assuming load_data() is a function that loads your dataset
    assert isinstance(data, pd.DataFrame), "The data is not a DataFrame"
    assert not data.empty, "The dataset is empty"
    assert 'target_column' in data.columns, "Target column is missing"  # Replace 'target_column' with your actual target column name

# Test preprocessing steps (like handling missing values)
def test_preprocess_data():
    data = load_data()
    preprocessed_data = preprocess_data(data)  # Assuming preprocess_data() handles missing values and preprocessing
    assert preprocessed_data.isnull().sum().sum() == 0, "Missing values exist after preprocessing"
    
    # Check if the types are optimized
    optimized_data = optimize_memory(preprocessed_data)  # Assuming optimize_memory optimizes the data types
    assert optimized_data.memory_usage(deep=True).sum() < preprocessed_data.memory_usage(deep=True).sum(), "Memory usage not optimized"

# Test for expected column types after preprocessing
def test_column_types():
    data = load_data()
    preprocessed_data = preprocess_data(data)
    assert preprocessed_data['feature_column'].dtype == 'float32', "Feature column type not optimized"  # Replace with actual feature column name
    assert preprocessed_data['target_column'].dtype == 'int32', "Target column type not optimized"  # Replace with actual target column name

# Test that no outliers exist (you can modify this based on your outlier detection logic)
def test_outlier_removal():
    data = load_data()
    # Assuming your preprocess_data function should handle outliers
    preprocessed_data = preprocess_data(data)
    # Implement your outlier detection check here (e.g., checking if values are within expected range)
    assert preprocessed_data['feature_column'].min() >= 0, "Outliers found in feature_column"  # Replace with actual logic

# Test class balance (you could modify this based on the way you handle class imbalance)
def test_class_balance():
    data = load_data()
    class_counts = data['target_column'].value_counts()
    assert abs(class_counts[0] - class_counts[1]) < class_counts.sum() * 0.1, "Class imbalance too large"  # Assuming binary classification

