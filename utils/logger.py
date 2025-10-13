import logging
import pathlib


def get_logger(name: str, log_file: pathlib.Path) -> logging.Logger:
    """
    Returns a logger that logs to a specific file and to console in case of
    warnings or higher concern.
    Separate loggers can be used for queries and evaluation.

    Args:
        name (str): The name of the logger.
        log_file (pathlib.Path): The path to the log file to write to.

    Returns:
        logging.Logger: The logger in question.
    """
    # Ensure logs/ dir exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Init logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Setup handlers
    if not logger.hasHandlers():
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # File handler
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Add handlers
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
