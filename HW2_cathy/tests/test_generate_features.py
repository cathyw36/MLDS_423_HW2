import pandas as pd
import pytest
from src.generate_features import generate_features

@pytest.fixture
def sample_data():
    """Fixture to provide sample DataFrame used in multiple tests."""
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}).astype(float)

@pytest.fixture
def basic_config():
    """Fixture to provide a basic configuration for feature engineering."""
    return {
        "feature_col": ["A", "B"],
        "target_col": "C",
        "feature_eng": [
            {"operation": "multiply", "source1": "A", "source2": "B", "target": "D"}
        ]
    }

# Happy Path Tests
def test_basic_operation(sample_data, basic_config):
    result = generate_features(sample_data, basic_config)
    expected_result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [4, 10, 18]}).astype(float)
    pd.testing.assert_frame_equal(result, expected_result)

def test_nested_operation(sample_data):
    config = {
        "feature_col": ["A", "B"],
        "target_col": "C",
        "feature_eng": [
            {
                "operation": "multiply",
                "source1": {"operation": "subtract", "source1": "A", "source2": "B"},
                "source2": "B",
                "target": "D",
            }
        ]
    }
    result = generate_features(sample_data, config)
    expected_result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [-12, -15, -18]}).astype(float)
    pd.testing.assert_frame_equal(result, expected_result)

def test_apply_operation(sample_data):
    config = {
        "feature_col": ["A", "B"],
        "target_col": "C",
        "feature_eng": [{"target": "D", "operation": "apply", "source1": "A", "function": "square"}]
    }
    result = generate_features(sample_data, config)
    expected_result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [1, 4, 9]}).astype(float)
    pd.testing.assert_frame_equal(result, expected_result)

# Unhappy Path Tests
def test_missing_feature_col(sample_data):
    config = {
        "feature_col": "A",  # Incorrect type: should be a list
        "target_col": "C",
        "feature_eng": [{"target": "D", "operation": "multiply", "source1": "A", "source2": "B"}]
    }
    with pytest.raises(KeyError):
        generate_features(sample_data, config)

def test_invalid_operation(sample_data):
    config = {
        "feature_col": ["A", "B"],
        "target_col": "C",
        "feature_eng": [{"target": "D", "operation": "invalid_operation", "source1": "A", "source2": "B"}]
    }
    with pytest.raises(NotImplementedError):
        generate_features(sample_data, config)

def test_invalid_function(sample_data):
    config = {
        "feature_col": ["A", "B"],
        "target_col": "C",
        "feature_eng": [{"target": "D", "operation": "apply", "source1": "A", "function": "invalid_function"}]
    }
    with pytest.raises(AttributeError):
        generate_features(sample_data, config)
