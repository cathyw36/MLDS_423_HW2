# Cloud Models

## Introduction

Welcome to the Cloud Models repository. This project encapsulates a structured and modularized approach to the machine learning development cycle, designed for ease of use and adaptability. The framework includes modules for each key step of the process, from initial data gathering and processing to the final stages of model evaluation and deployment.

## Contents

- [Cloud Models](#cloud-models)
  - [Introduction](#introduction)
  - [Contents](#contents)
  - [Key Features](#key-features)
  - [Prerequisites](#prerequisites)
  - [Getting Started](#getting-started)
    - [Repository Cloning](#repository-cloning)
    - [Navigating to the Repository](#navigating-to-the-repository)
    - [Configuring AWS for S3 Uploads](#configuring-aws-for-s3-uploads)
    - [Dependencies Installation](#dependencies-installation)
  - [Adjustments and Enhancements of config file](#adjustments-and-enhancements)
    - [Data Collection Adjustments](#data-collection-adjustments)
    - [Dataset Configuration](#dataset-configuration)
    - [Feature Configuration](#feature-configuration)
    - [Analytical Adjustments](#analytical-adjustments)
    - [Model Training Adjustments](#model-training-adjustments)
    - [Model Scoring Adjustments](#model-scoring-adjustments)
    - [Performance Evaluation Adjustments](#performance-evaluation-adjustments)
    - [AWS Configuration](#aws-configuration)
  - [Operational Guide](#operational-guide)
    - [Execution Locally](#execution-locally)
      - [Running the Pipeline](#running-the-pipeline)
      - [Executing Unit Tests](#executing-unit-tests)
    - [Using Docker](#using-docker)
      - [Pipeline Execution](#pipeline-execution)
        - [Docker Image Construction](#docker-image-construction)
        - [Launching the Pipeline](#launching-the-pipeline)
      - [Testing with Docker](#testing-with-docker)
        - [Constructing the Docker Image for Testing](#constructing-the-docker-image-for-testing)
        - [Initiating the Tests](#initiating-the-tests)
  

## Key Features

- **Modularity:** The structure is built with self-contained modules for each phase of the machine learning pipeline.
- **Customizability:** Easy adjustments are possible through the [config.yaml](config/config.yaml) to meet specific requirements.
- **Broad Compatibility:** Supports various libraries such as TensorFlow, PyTorch, and scikit-learn.
- **Reproducibility:** Ensures consistent results through Docker encapsulation.

## Prerequisites

- Python 3.7+
- Dependencies listed in [requirements.txt](requirements.txt).

## Getting Started

### Repository Cloning

```bash
git clone https://github.com/cathyw36/MLDS_423_HW2.git
```

### Navigating to the Repository

```bash
cd HW2_Cathy
```

### Configuring AWS for S3 Uploads

Ensure the `AWS CLI` is installed. Configure your AWS profile as follows:
During the configure, you need to get your key from AWS IAM serives, with ID and Secret Key
```bash
aws configure 
```

### Dependencies Installation

```bash
pip install -r requirements.txt
```

## Adjustments and Enhancements of config

This section outlines how you can customize and enhance various components of the cloud classification pipeline according to your specific needs. Each sub-section corresponds to key areas in the `config.yaml` file that can be modified for different use cases.

### Data Collection Adjustments

To adapt the data collection process to your needs, modify the `run_config` section. Here, you can change the data source or the output directory structure as required:

```yaml
run_config:
  data_source: "https://newsource.com/newdata"
  data_dir:
    raw: new_data/raw
    processed: new_data/processed
```

### Dataset Configuration

Adjust the dataset creation parameters in the `create_dataset` section to suit different data formats or preprocessing needs:

```yaml
create_dataset:
  date_config:
    date_format: "%d-%m-%Y"  # Adjust date formats as per new requirements
  data_prep:
    first_cloud:
      left: 50  # Adjust indices for new data structure
      right: 1100
```

### Feature Engineering

Modify the `generate_features` section to introduce new features, adjust existing feature calculations, or redefine the target variable for classification:

```yaml
generate_features:
  feature_col:
    - new_feature1
    - new_feature2
  feature_eng:
    - operation: new_operation
      source1: new_feature1
      source2: new_feature2
      target: new_feature3
```

### Matplotlib Configuration

Tweak `mpl_config` for aesthetic adjustments or to accommodate different visualization requirements:

```yaml
mpl_config:
  font.size: 18
  axes.labelsize: 22
  figure.figsize: [14.0, 10.0]
```

### Model Training Adjustments

Customize the `train_model` section to change the machine learning model, adjust hyperparameters, or modify the train-test split:

```yaml
train_model:
  model_config:
    type: NewModelType
    hyperparam:
      new_param1: value1
      new_param2: value2
```

### Model Scoring and Evaluation

Update the `score_model` and `evaluate_performance` sections to alter scoring metrics or the way model performance is evaluated:

```yaml
score_model:
  initial_features:
    - adjusted_feature1
    - adjusted_feature2

evaluate_performance:
  metrics:
    - new_metric1
    - new_metric2
```

### AWS Configuration

If using AWS for deployments or data handling, adjust the `aws` settings to change the storage bucket or manage permissions:

```yaml
aws:
  bucket_name: new-bucket-name
  prefix: new-prefix/
```

Each of these adjustments allows you to optimize the pipeline for different datasets, operational environments, or project requirements, ensuring flexibility and scalability of your machine learning operations.



## Operational Guide

### Execution Locally

#### Running the Pipeline

Ensure you are in the same directory as `pipeline.py` and execute:

```bash
python pipeline.py
```

#### Executing Unit Tests

Run the tests using:

```bash
pytest
```

### Using Docker

#### Pipeline Execution

##### Docker Image Construction

```bash
docker build -t pipeline -f dockerfiles/dockerfile-pipeline .
```

##### Launching the Pipeline

```bash
docker run -v ~/.aws:/root/.aws -e AWS_PROFILE=my-sso pipeline
```

#### Testing with Docker

##### Constructing the Docker Image for Testing

```bash
docker build -t unittest-pipeline -f dockerfiles/dockerfile-test .
```

##### Initiating the Tests

```bash
docker run unittest-pipeline
```
