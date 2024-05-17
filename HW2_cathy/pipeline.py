import argparse
import datetime
import logging.config
from pathlib import Path
import yaml
import botocore

# Assuming all these are implemented and available under the src directory.
import src.acquire_data as ad
import src.analysis as eda
import src.aws_utils as aws
import src.create_dataset as cd
import src.evaluate_performance as ep
import src.generate_features as gf
import src.score_model as sm
import src.train_model as tm

import logging

# Setup logging as configured in local.conf
logging.config.fileConfig('config/logging/local.conf', disable_existing_loggers=False)

# Adjusting logging levels for external libraries
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger("clouds")
logger.info("Logging is configured.")

def setup_logging():
    """ Setup logging configuration. """
    logging.config.fileConfig("config/logging/local.conf", disable_existing_loggers=False)
    return logging.getLogger("clouds")

def load_config(path):
    """ Load configuration from YAML file. """
    try:
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        logger.error(f"Error loading configuration from {path}: {e}")
        exit(1)

def create_directories(base_path, config):
    """ Create directories based on configuration settings. """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_path = Path(base_path) / f"run_{now}"
    artifacts_path.mkdir(parents=True, exist_ok=True)

    raw_data_dir = artifacts_path / Path(config["run_config"]["data_dir"]["raw"])
    processed_data_dir = artifacts_path / Path(config["run_config"]["data_dir"]["processed"])
    figure_dir = artifacts_path / Path(config["run_config"]["figure_dir"])
    model_data_dir = artifacts_path / Path(config["train_model"]["data_dir"])
    model_dir = artifacts_path / Path(config["train_model"]["model_dir"])
    score_dir = artifacts_path / Path(config["score_model"]["score_dir"])
    metric_dir = artifacts_path / Path(config["evaluate_performance"]["metric_dir"])

    for dir in [raw_data_dir, processed_data_dir, figure_dir, model_data_dir, model_dir, score_dir, metric_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    return raw_data_dir, processed_data_dir, figure_dir, model_data_dir, model_dir, score_dir, metric_dir, artifacts_path

def main(config_path):
    """ Main execution function. """
    logger = setup_logging()
    config = load_config(config_path)

    base_path = config["run_config"]["output"]["runs"]
    raw_data_dir, processed_data_dir, figure_dir, model_data_dir, model_dir, score_dir, metric_dir, artifacts_path = create_directories(base_path, config)

    ad.acquire_data(config["run_config"]["data_source"], raw_data_dir / "clouds.data")

    data = cd.create_dataset(raw_data_dir / "clouds.data", config["create_dataset"])
    cd.save_dataset(data, processed_data_dir / "clouds.csv")

    features = gf.generate_features(data, config["generate_features"])
    eda.save_figures(features, figure_dir, config)

    model, train, test = tm.train_model(features, config["train_model"])
    tm.save_data(train, test, model_data_dir)
    tm.save_model(model, model_dir / "trained_model_object.pkl")

    scores = sm.score_model(test, model, config["score_model"])
    sm.save_scores(scores, score_dir / "scores.csv")

    metrics = ep.evaluate_performance(test, scores, config["evaluate_performance"])
    ep.save_metrics(metrics, metric_dir / "metrics.yaml")

    if config["aws"].get("upload", False):
        aws.upload_artifacts(artifacts_path, config["aws"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full pipeline for model training and evaluation.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    main(args.config)
