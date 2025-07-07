
#Run this notebook to run the full AirBnB price prediction model

import sys
import os
sys.path.append(os.path.abspath(""))

import pandas as pd 
import torch


# Import functions from src modules
from image_predictions import generate_image_predictions
from random_forest_with_tuning import run_random_forest

# Configuration
LISTINGS_CSV = "../data/listings.csv"
IMAGE_DIR = "../data/images"
MODEL_PATH = "../models/lenet5.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## create image predictions
## @Chris make this possible!
print("Generating predictions from image model...")
image_predictions = generate_image_predictions(
    csv_file=LISTINGS_CSV,
    image_dir=IMAGE_DIR,
    model_path=MODEL_PATH,
    device=DEVICE   
)


# Run full tabular pipeline including iamge features
print("Running tabular pipeline with image features...")
rune_tabular_pipeline(
    csv_file=LISTINGS_CSV,
    image_predictions=image_predictions
)

print("Full pipeline completed successfully!")
