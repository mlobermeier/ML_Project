import sys
import os
# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas
import torch


# Import functions from src modules
from image_predictions import generate_image_predictions
from random_forest_model import run_random_forest

# Configuration
LISTINGS_CSV = "../data/listings.csv"
IMAGE_DIR = "../data/images"
MODEL_PATH = "../models/lenet5.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## create image predictions
print("Generating predictions from image model...")
image_predictions = generate_image_predictions(
    csv_file=LISTINGS_CSV,
    image_dir=IMAGE_DIR,
    model_path=MODEL_PATH,
    device=DEVICE   
)


# Run full tabular pipeline including iamge features
print("Running tabular pipeline with image features...")
model, metrics, preds, figs = run_random_forest(
    csv_file=LISTINGS_CSV,
    image_predictions=image_predictions
)

print("MAE:", metrics["MAE"])
print("RMSE:", metrics["RMSE"])

for name, fig in figs.items():
    fig.savefig(f"../figures/{name}.png")
    print(f"Saved figure: {name}.png")

print("Full pipeline completed successfully!")
