import os
from pathlib import Path

PROJ_CODE = 'BE'

BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
PROJ_DIR = BASE_DIR

CACHE_DIR = PROJ_DIR / 'cache'
BATCH_DIR = PROJ_DIR / 'batch'
DATA_DIR = PROJ_DIR / 'data'
DATASET_DIR = PROJ_DIR / 'dataset'
MODEL_DIR = PROJ_DIR / 'model'
OT2_PROC_DIR = PROJ_DIR / 'ot2_proc'

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OT2_PROC_DIR, exist_ok=True)
