import logging


def setup_logger(name='projet_deep_learning'):
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
