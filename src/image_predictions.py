import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.pardir, "src")))
from data_loader import image_downloader
from image_lenet import ListingsDataset
from image_lenet import LeNet5

#########################################################
def run_predictions(csv_file, img_dir, device_, model_path):
    #load the dataset and save image urls and listings ids in a dictionary
    #DONT RUN THIS UNLESS YOU WANT TO DOWNLOAD ALL IMAGES IT TAKES A LONG TIME
    df = pd.read_csv(csv_file)
    id_image_dicct = {}
    for index, row in df.iterrows():
        if pd.notna(row["picture_url"]):
            id_image_dicct[row["id"]] = row["picture_url"]
    num_images = len(id_image_dicct)
    print(f"Total images to download: {num_images}")
    #download images
    #for listing_id, image_url in id_image_dicct.items():
    #    image_downloader(image_url, listing_id, True)

    ############################################################

    # set up transforms and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalization if needed
    ])

    dataset = ListingsDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    #############################################################

    # Device is defined in pipeline, could be changed here if needed
    device = torch.device(device_)

    ###########################################################

    model = LeNet5().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, prices in train_loader:
            images = images.to(device)
            prices = prices.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, prices)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    ######################################################

    # Get a sample from the validation set
    val_img, val_price = val_dataset[0]
    val_img = val_img.unsqueeze(0).to(device)  # Add batch dimension and move to device

    model.eval()
    with torch.no_grad():
        pred_price = model(val_img)
        print(f"Predicted price: {pred_price.item():.2f}, Actual price: {val_price:.2f}")

    ######################################################

    # plot all predictions vs actual prices on the validation set
    model.eval()
    predicted_prices = []
    actual_prices = []

    with torch.no_grad():
        for images, prices in val_loader:
            images = images.to(device)
            prices = prices.float().unsqueeze(1).to(device)
            outputs = model(images)
            predicted_prices.extend(outputs.cpu().numpy().flatten())
            actual_prices.extend(prices.cpu().numpy().flatten())

    fig_pred_vs_price, ax_pr = plt.subplots(figsize=(10, 6))
    ax_pr.plot(actual_prices, label='Actual Prices', marker='o', linestyle='-', alpha=0.7)
    ax_pr.plot(predicted_prices, label='Predicted Prices', marker='x', linestyle='-', alpha=0.7)
    ax_pr.set_xlabel('Sample Index')
    ax_pr.set_ylabel('Price')
    ax_pr.set_title('Actual vs Predicted Prices on Validation Set')
    ax_pr.legend()
    ax_pr.grid(True)

    ###########################################################

    # Use classification mode with default bins
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalization if needed
    ])

    # Create dataset in classification mode
    dataset = ListingsDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=transform,
        target_type="classification"
    )

    num_classes = dataset.data["price_class"].nunique()
    print(f"Number of classes: {num_classes}")

    # Split into train and validation sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer
    device = torch.device(device_)
    model = LeNet5(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    epoch_losses_classification = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)  # CrossEntropyLoss expects LongTensor

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses_classification.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Evaluate accuracy on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2%}")

    ############################################################

    # Pick a sample from the validation set
    sample_img, sample_label = val_dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        output = model(sample_img)
        pred_class = output.argmax(dim=1).item()

    # Get the original index in the dataset
    orig_idx = val_dataset.indices[0]
    row = df.iloc[orig_idx]
    real_price = row["price"]
    image_url = row["picture_url"]

    print(f"Predicted price class: {pred_class}")
    print(f"Actual price: {real_price}")
    print(f"Image URL: {image_url}")

    ###########################################################
    # Save the model
    torch.save(model.state_dict(), model_path)

    log = {"total_images_downloaded": num_images,
           "epoch_losses": epoch_losses,
           "number_of_classes": num_classes,
           "epoch_losses_classification": epoch_losses_classification,
            "validation_accuracy": accuracy,
           }
    
    figs = {
        "fig_pred_vs_price": fig_pred_vs_price
    }

    return {
        "image_predictions": image_predictions,
        "model": model,
        "log": log,
        "figs": figs
    }
