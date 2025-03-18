import pytest
import pandas as pd
from src.utils.data_processing import optimize_memory

# Test if the memory optimization function works as expected
def test_optimize_memory():
    # Create a dummy dataframe with large data types
    data = {
        'int_column': [1, 2, 3, 4, 5],
        'float_column': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_column': ['a', 'b', 'c', 'd', 'e']
    }

    df = pd.DataFrame(data)

    # Check initial memory usage
    initial_memory = df.memory_usage(deep=True).sum()

    # Apply memory optimization
    optimized_df = optimize_memory(df)

    # Check memory usage after optimization
    optimized_memory = optimized_df.memory_usage(deep=True).sum()

    # Assert that optimized memory usage is less than initial memory usage
    assert optimized_memory < initial_memory, "Memory usage was not optimized"

# Test if the dataframe has no missing values after preprocessing (assuming preprocess_data handles missing values)
def test_missing_values_handling():
    # Create a dataframe with missing values
    data_with_missing = {
        'feature1': [1, 2, 3, None, 5],
        'feature2': [None, 2, 3, 4, 5],
    }

    df = pd.DataFrame(data_with_missing)

    # Assume preprocess_data handles missing values
    preprocessed_df = preprocess_data(df)

    # Assert that there are no missing values after preprocessing
    assert preprocessed_df.isnull().sum().sum() == 0, "Missing values still exist after preprocessing"

# Test if the dataframe after preprocessing has the expected column types
def test_column_types():
    data = {
        'int_column': [1, 2, 3, 4, 5],
        'float_column': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_column': ['a', 'b', 'c', 'd', 'e']
    }

    df = pd.DataFrame(data)

    # Apply memory optimization
    optimized_df = optimize_memory(df)

    # Check column types after optimization
    assert optimized_df['int_column'].dtype == 'int32', "int_column type was not optimized"
    assert optimized_df['float_column'].dtype == 'float32', "float_column type was not optimized"
    assert optimized_df['str_column'].dtype == 'object', "str_column type was not optimized"

# Test if outliers are removed (assuming preprocess_data handles outliers)
def test_outlier_removal():
    data_with_outliers = {
        'feature': [1, 2, 3, 100, 5]
    }

    df = pd.DataFrame(data_with_outliers)

    # Apply preprocessing (assuming it handles outliers)
    processed_df = preprocess_data(df)

    # Assert that outliers are handled (e.g., no values above a threshold)
    assert processed_df['feature'].max() <= 10, "Outliers were not removed"

# Test class balance (if needed based on the imbalance handling strategy)
def test_class_balance():
    # Simulate a dataset with class imbalance
    data = {
        'feature': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'target': [0, 0, 0, 1, 1, 0, 1, 0, 0]
    }

    df = pd.DataFrame(data)

    # Get class counts
    class_counts = df['target'].value_counts()

    # Check if the class imbalance is handled appropriately
    assert abs(class_counts[0] - class_counts[1]) < class_counts.sum() * 0.1, "Class imbalance is too high"
