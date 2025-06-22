# function to download the image for a certain airbnb listing id
def image_downloader(url: str, id: str, resize: bool) -> None:
    import requests
    from pathlib import Path

    local_dir = Path("../data/image_per_listing")
    local_dir.mkdir(parents=True, exist_ok=True)
    image_path = local_dir / f"{id}.jpg"

    # Skip download if file already exists
    if image_path.exists():
        return

    # Download the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(image_path, "wb") as file:
            file.write(response.content)
        # Resize the image if required
        if resize:
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                with Image.open(image_path) as img:
                    img = img.resize((32, 32))
                    img = img.convert("RGB")  # Ensure the image is in RGB format
                    img.save(image_path)
            except Exception as e:
                print(f"Skipping {id}: {e}")
                image_path.unlink(missing_ok=True)  # Remove the bad image file
    else:
        print(f"Failed to download image for {id}. Status code: {response.status_code}")