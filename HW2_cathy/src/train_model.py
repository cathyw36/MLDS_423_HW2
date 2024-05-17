from importlib import import_module
from pathlib import Path
import os
import pickle
import logging
import pandas as pd
import sklearn

logger = logging.getLogger("clouds")


def train_model(data: pd.DataFrame, config: dict) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """Train a model based on configuration settings.

    Args:
        data (pd.DataFrame): Data for the model to be trained on.
        config (dict): Configuration for model training including feature selection and hyperparameters.

    Returns:
        A tuple containing the trained model, and the train and test dataframes.
    """
    features = data[config["initial_features"]]
    target = data[config["target"]]
    model_config = config["model_config"]

    # Dynamic library and model import based on config
    model_lib = import_module(model_config["model_lib"])
    model_class = getattr(model_lib, model_config["type"])
    model = model_class(**model_config["hyperparam"])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, target, test_size=config["train_test_split"]["test_size"]
    )

    model.fit(x_train, y_train)
    logger.info("Model successfully trained")

    # Create DataFrame for train and test sets including target
    train = pd.DataFrame(x_train, columns=config["initial_features"])
    train[config["target"]] = y_train

    test = pd.DataFrame(x_test, columns=config["initial_features"])
    test[config["target"]] = y_test

    return model, train, test


def save_data(train: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    """Save train and test datasets to specified directory.

    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
        path (Path): Path to save datasets.
    """
    train_path = path / "train.csv"
    test_path = path / "test.csv"

    train.to_csv(train_path, index=False)
    logger.info("Train dataset successfully saved to %s", train_path)

    test.to_csv(test_path, index=False)
    logger.info("Test dataset successfully saved to %s", test_path)


def save_model(model: object, path: Path) -> None:
    """Save a trained model to a specified path.

    Args:
        model (object): Trained model.
        path (Path): Path to save the model binary.
    """
    with open(path, "wb") as file:
        pickle.dump(model, file)
        logger.info("Model binary successfully saved to %s", path)
