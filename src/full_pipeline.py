import sys
import os
# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas
import torch

# Import functions from src modules
from image_predictions import generate_image_predictions
from boosting_test import run_boosting

import os
from pathlib import Path

# Get absolute path to the current script
current_path = Path(__file__).resolve()

# Search upward for ML_project
for parent in current_path.parents:
    if parent.name == "ML_project":
        os.chdir(parent)
        print(f"Working directory set to: {parent}")
        break
else:
    raise FileNotFoundError("Could not find 'ML_project' in the path hierarchy.")


# Configuration
LISTINGS_CSV = os.path.join("data", "listings.csv") 
IMAGE_DIR = os.path.join("data", "images")
MODEL_PATH = os.path.join("models", "lenet5.pth")  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


## create image predictions
print("Generating predictions from image model...")
image_outputs = generate_image_predictions(
    csv_file=LISTINGS_CSV,
    img_dir=IMAGE_DIR,
    model_path=MODEL_PATH,
    device_=DEVICE   
)

image_predictions = image_outputs["image_predictions"]
img_model = image_outputs["model"]
img_log = image_outputs["log"]
img_figs = image_outputs["figs"]


# Run full tabular pipeline including iamge features
print("Running tabular pipeline with image features...")
boosting_output = run_boosting(
    csv_file=LISTINGS_CSV,
    image_predictions=image_predictions
)

boosting_model = boosting_output["model"]
boosting_metrics = boosting_output["metrics"]
boosting_preds = boosting_output["preds"]
boosting_figs = boosting_output["figs"]
boosting_log = boosting_output["log"]
boosting_features = boosting_output["features"]

print("MAE:", boosting_metrics["MAE"])
print("RMSE:", boosting_metrics["Validation RMSE"])

for name, fig in boosting_figs.items():
    fig_path = os.path.join("figures", f"{name}.png")
    fig.savefig(fig_path)
    print(f"Saved figure: {name}.png")

print("Full pipeline completed successfully!")
