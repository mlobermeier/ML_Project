import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class ListingsDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_type="regression", bins=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_type = target_type

        # Clean the price column: remove $ and commas, convert to float, and drop NaN
        self.data["price"] = (
            self.data["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
        self.data = self.data.dropna(subset=["price"]).reset_index(drop=True)

        #create price bins if target_type is classification
        if target_type == "classification":
            if bins is None:
                #default bins
                bins = [0, 50, 100, 150, 200, 500, 10000]
            self.data["price_class"] = pd.cut(self.data["price"], bins=bins, labels=False, include_lowest=True)
            self.data = self.data.dropna(subset=["price_class"]).reset_index(drop=True)  # <--- ADD THIS LINE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import os
        from PIL import Image

        img_name = os.path.join(self.img_dir, f"{self.data.iloc[idx, 0]}.jpg")
        try:
            if not os.path.exists(img_name):
                raise FileNotFoundError
            image = Image.open(img_name)
            image = image.convert("RGB")
        except (FileNotFoundError, OSError) as e:
            # OSError will catch truncated images and other image loading errors
            next_idx = idx + 1
            while next_idx < len(self.data):
                next_img_name = os.path.join(self.img_dir, f"{self.data.iloc[next_idx, 0]}.jpg")
                try:
                    if os.path.exists(next_img_name):
                        image = Image.open(next_img_name)
                        image = image.convert("RGB")
                        idx = next_idx
                        break
                except OSError:
                    pass  # Try next image if this one is also corrupted/truncated
                next_idx += 1
            else:
                # If no next image found, raise IndexError
                raise IndexError(f"No available image found after index {idx}")
        if self.transform:
            image = self.transform(image)
        if self.target_type == "regression":
            target = self.data.iloc[idx]['price']
        else:  # classification
            target = int(self.data.iloc[idx]['price_class'])
        return image, target

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        if num_classes == None:
            self.fc3 = nn.Linear(84, 1)  # Regression: 1 output
        else:
            self.fc3 = nn.Linear(84, num_classes) # Classification: num_classes outputs

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x