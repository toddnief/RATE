import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR.parent / ".env")

# Note: Make sure these are set in the .env file
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
GROUP_NAME = os.getenv("GROUP_NAME")

DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HH_RLHF_PATH = DATA_DIR / "hh-rlhf" / "hh-rlhf.json"
ELI_5 = DATA_DIR / "ELI5" / "all.jsonl"

API_DIR = DATA_DIR / "batch_api"
API_DIR.mkdir(parents=True, exist_ok=True)

REWRITES_DIR = DATA_DIR / "rewrites"
REWRITES_DIR.mkdir(parents=True, exist_ok=True)

SCORED_DIR = DATA_DIR / "scored"
SCORED_DIR.mkdir(parents=True, exist_ok=True)

EFFECTS_DIR = PROJECT_DIR / "effects"
EFFECTS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = PROJECT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FILE_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize config constants to None — will be lazy loaded in load_config
# Make sure to import and run load_config in scripts that use these constants
SMOKE_TEST = None
REWRITES_DATASET_NAME = None


def load_config(config_path=None):
    config_file = config_path if config_path else SCRIPT_DIR / "config.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    global SMOKE_TEST, REWRITES_DATASET_NAME

    # Set constants based on config
    SMOKE_TEST = config["smoke_test"]
    REWRITES_DATASET_NAME = config["rewrites"]["dataset_name"]

    logging.info(f"Loaded config from {config_file}")
