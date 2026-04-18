import logging
import sys
import numpy as np

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

def save_npz(filepath, data_dict):
    """
    Save dictionary to npz file
    """
    np.savez_compressed(filepath, **data_dict)

def load_npz(filepath):
    """
    Load dictionary from npz file
    """
    return dict(np.load(filepath, allow_pickle=True))
