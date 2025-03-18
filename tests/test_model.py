test_model
import pytest
from src.models import train_model

def test_train_model():
    # Mock input data
    X = [[0, 1], [1, 0]]
    y = [0, 1]

    # Train the model
    model = train_model(X, y)

    # Check if model is trained
    assert model is not None

