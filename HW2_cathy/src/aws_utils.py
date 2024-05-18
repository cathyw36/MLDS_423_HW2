import os
import logging
from typing import List
from pathlib import Path
import boto3

# Configure logging
logger = logging.getLogger("clouds")

def check_bucket_exists(bucket_name: str, s3_client):
    """Verify the existence of an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_client: The S3 client instance used to communicate with AWS S3.

    Raises:
        NotImplementedError: If bucket does not exist or other S3 errors occur.
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info("Bucket '%s' verified.", bucket_name)
    except Exception as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.error("Bucket '%s' does not exist.", bucket_name)
            raise NotImplementedError("Bucket does not exist.") from e
        else:
            logger.error("Error checking bucket '%s': %s", bucket_name, e)
            raise NotImplementedError from e

def upload_artifacts(artifacts: Path, config: dict) -> List[str]:
    """Uploads local files as artifacts to an AWS S3 bucket.

    Args:
        artifacts (Path): The local directory containing files to upload.
        config (dict): Configuration including the bucket name and prefix.

    Returns:
        List[str]: A list of S3 URIs to the uploaded artifacts.

    Raises:
        NotImplementedError: If the upload fails.
    """
    s3_client = boto3.client("s3")
    bucket_name = config["bucket_name"]
    prefix = config.get("prefix", "")  # Using .get for safer access

    check_bucket_exists(bucket_name, s3_client)

    s3_uris = []
    for root, _, files in os.walk(artifacts):
        for file in files:
            local_file_path = Path(root) / file
            s3_key = local_file_path.relative_to(artifacts).as_posix()
            s3_uri = f"s3://{bucket_name}/{prefix}/{s3_key}"

            try:
                with open(local_file_path, "rb") as file_data:
                    s3_client.put_object(Bucket=bucket_name, Key=f"{prefix}/{s3_key}", Body=file_data)
                    logger.info("Successfully uploaded '%s' to '%s'.", local_file_path, s3_uri)
            except Exception as e:
                logger.error("Failed to upload '%s' to '%s': %s", local_file_path, s3_uri, e)
                raise NotImplementedError from e

            s3_uris.append(s3_uri)

    return s3_uris
