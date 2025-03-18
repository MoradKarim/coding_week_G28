import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to load the dataset
def load_data(data_path):
    """
    Load dataset from the specified path.
    """
    df = pd.read_csv(data_path)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Function to preprocess the data
def preprocess_data(df):
    """
    Preprocess the data by handling missing values, scaling numerical columns,
    and optimizing memory usage.
    """
    # Handle missing values
    df = df.fillna(df.mean())  # Simple strategy: Fill missing values with column means
    print(f"Missing values handled. {df.isnull().sum().sum()} missing values left.")
    
    # Remove irrelevant or highly correlated features if needed
    # Example: Drop unnecessary columns
    df.drop(columns=['unnecessary_column'], inplace=True)  # Replace with actual column name if needed
    
    # Handle class imbalance if applicable
    # This might be done later during model training or by resampling (e.g., SMOTE)
    
    # Optimize memory usage
    df = optimize_memory(df)
    
    # Example: Feature scaling for numerical columns (optional)
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df


