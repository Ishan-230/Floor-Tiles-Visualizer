"""
finetune_deeplabv3.py (with BlendedData integration + visualization)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


# --------------------------
# Dataset Class
# --------------------------
class CorridorDataset(Dataset):
    """Custom dataset for corridor images and ground truth masks."""
    def __init__(self, image_paths, mask_paths, resize_size=(520, 520)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.resize_size = resize_size

        self.img_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(resize_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.resize_size, interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 127, 1, 0).astype(np.int64)

        img = self.img_transform(img)
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask


# --------------------------
# Visualization function
# --------------------------
def visualize_predictions(model, dataset, device, num_samples=3):
    model.eval()
    idxs = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axs = plt.subplots(num_samples, 3, figsize=(10, 4 * num_samples))

    for i, idx in enumerate(idxs):
        img, mask = dataset[idx]
        img_in = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_in)["out"]
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        # Denormalize image for display
        img_disp = img.permute(1, 2, 0).cpu().numpy()
        img_disp = (img_disp * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
        img_disp = np.clip(img_disp, 0, 1)

        axs[i, 0].imshow(img_disp)
        axs[i, 0].set_title("Input Image")
        axs[i, 1].imshow(mask.cpu().numpy(), cmap="gray")
        axs[i, 1].set_title("Ground Truth")
        axs[i, 2].imshow(pred, cmap="gray")
        axs[i, 2].set_title("Prediction")

        for j in range(3):
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------
# Training Script
# --------------------------
def main():
    base_dir = "E:/College/Project/Floor Tiles Visualizer/corridor"
    raw_dir = os.path.join(base_dir, "raw_image")
    mask_dir = os.path.join(base_dir, "ground_truth")
    blended_dir = os.path.join(base_dir, "BlendedData")

    raw_images = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    raw_masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    assert len(raw_images) == len(raw_masks), "Raw image/mask counts mismatch!"

    train_images, val_images, train_masks, val_masks = train_test_split(
        raw_images, raw_masks, test_size=0.2, random_state=42
    )

    blended_images = sorted([os.path.join(blended_dir, f) for f in os.listdir(blended_dir)
                             if f.lower().endswith(('.jpg', '.png', '.jpeg')) and "mask" not in f.lower()])
    blended_masks = sorted([os.path.join(blended_dir, f) for f in os.listdir(blended_dir)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg')) and "mask" in f.lower()])

    assert len(blended_images) == len(blended_masks), "Blended image/mask counts mismatch!"

    train_images += blended_images
    train_masks += blended_masks

    print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")

    train_dataset = CorridorDataset(train_images, train_masks, resize_size=(520, 520))
    val_dataset = CorridorDataset(val_images, val_masks, resize_size=(520, 520))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss/len(train_dataset):.4f}, "
              f"Val Loss: {val_loss/len(val_dataset):.4f}")

        # 🔍 Show predictions after each epoch
        visualize_predictions(model, val_dataset, device, num_samples=2)

    save_path = os.path.join(base_dir, "deeplabv3_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved at {save_path}")


if __name__ == "__main__":
    main()
