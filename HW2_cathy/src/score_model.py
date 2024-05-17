from pathlib import Path
from typing import Tuple
import logging
import pandas as pd

logger = logging.getLogger("clouds")


def score_model(test: pd.DataFrame, model: object, config: dict) -> Tuple[list, list]:
    """
    Score a saved model using test data.

    Args:
        test (pd.DataFrame): Test dataset.
        model (object): Trained model to score.
        config (dict): Configurations for scoring the model including feature selection.

    Returns:
        Tuple[list, list]: A tuple containing the predicted probabilities of the positive class and the predicted classes.
    """
    initial_features = config["initial_features"]
    x_test = test[initial_features]

    ypred_proba_test = model.predict_proba(x_test)[:, 1]  # Probabilities of the positive class
    ypred_bin_test = model.predict(x_test)               # Binary predictions
    logger.info("Model predictions (probability and class) created.")

    return ypred_proba_test, ypred_bin_test  # Parentheses are optional


def save_scores(scores: Tuple[list, list], path: Path) -> None:
    """
    Save the output of the model to a CSV file.

    Args:
        scores (Tuple[list, list]): Predicted scores of the model, including probabilities and classes.
        path (Path): Path to save model outputs.

    Raises:
        Exception: Raises an exception if saving the predictions to CSV fails.
    """
    pred_prob, pred_class = scores  # Unpacking for clarity
    out = pd.DataFrame({"Probability": pred_prob, "Class": pred_class})
    try:
        out.to_csv(path, index=False)  # Optionally avoid saving DataFrame index if not required
        logger.info("Model predictions saved to %s", path)
    except Exception as e:
        logger.error("Model predictions failed to save to %s due to %s", path, e)
        raise Exception("Error saving model predictions: {}".format(e)) from e

