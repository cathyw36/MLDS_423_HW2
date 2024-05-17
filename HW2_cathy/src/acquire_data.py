import logging
from pathlib import Path
import sys
import time

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clouds")

def get_data(url: str, retries: int = 4, base_delay: int = 3, delay_factor: int = 2) -> bytes:
    """Fetches data from a specified URL with retries and exponential backoff.

    Parameters:
        url (str): URL to fetch data from.
        retries (int): Maximum number of retry attempts. Defaults to 4.
        base_delay (int): Base delay between retries in seconds. Defaults to 3.
        delay_factor (int): Multiplicative factor for delay. Defaults to 2.

    Returns:
        bytes: Retrieved data.

    Raises:
        NotImplementedError: If all attempts to fetch data fail.
    """
    delay = base_delay
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Checks for HTTP error responses
            return response.content
        except requests.exceptions.RequestException as error:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed: {error}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= delay_factor
            else:
                logger.error("All attempts to retrieve data have failed.")
                raise NotImplementedError from error

def write_data(data: bytes, destination: Path) -> None:
    """Writes data to the specified file path.

    Parameters:
        data (bytes): Data to write.
        destination (Path): File path where data will be saved.

    Raises:
        NotImplementedError: If data writing fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(destination, 'wb') as file:
            file.write(data)
        logger.info(f"Data successfully written to {destination}")
    except Exception as exc:
        logger.error(f"Failed to write data: {exc}")
        raise NotImplementedError from exc

def acquire_data(url: str, file_path: Path) -> None:
    """Orchestrates the data fetching and writing process.

    Parameters:
        url (str): URL of the data source.
        file_path (Path): Path to save the fetched data.

    Raises:
        SystemExit: If there is an error related to file operations.
    """
    try:
        data = get_data(url)
        write_data(data, file_path)
    except FileNotFoundError:
        logger.error("Invalid file path provided.")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Error during file operations: {exc}")
        sys.exit(1)
