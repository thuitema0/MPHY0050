import os
import random
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


def set_seed(seed=42):
    """
    Set the random seed so results are more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def segment_and_crop(image_rgb):
    """
    Segment the retinal field of view and crop the image to that region.

    This is a simple preprocessing step to remove the black background
    around the fundus image. It does not segment lesions.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Threshold the image so the retinal area stays white
    _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    # Keep only the largest connected region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Clean the mask a little
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the original RGB image
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Crop to the mask bounding box
    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        cropped = segmented[y1:y2 + 1, x1:x2 + 1]
    else:
        cropped = segmented

    return cropped, mask


class MyopiaDataset(Dataset):
    """
    Dataset for myopic maculopathy classification.

    Each image is first segmented and cropped, then transformed before
    being passed to the model.
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = row["image"]
        label = int(row["myopic_maculopathy_grade"])

        image_path = os.path.join(self.img_dir, image_name)
        image_rgb = np.array(Image.open(image_path).convert("RGB"))

        cropped_rgb, _ = segment_and_crop(image_rgb)

        if self.transform is not None:
            image_tensor = self.transform(cropped_rgb)
        else:
            image_tensor = transforms.ToTensor()(cropped_rgb)

        return image_tensor, label

##### Models ####
def build_model(num_classes=5):
    """
    Build a ResNet18 model and replace the final layer
    so it matches the 5-class classification task.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# def build_model(num_classes=5):
#     """
#     Create the DenseNet121 model and replace the final layer.
#     """
#     model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
#     model.classifier = nn.Linear(model.classifier.in_features, num_classes)
#     return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation or test data.
    """
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds


def show_examples(df, img_dir, n=4):
    """
    Show a few examples of the original image, mask and cropped result.
    """
    sample_df = df.sample(n=n, random_state=42).reset_index(drop=True)

    plt.figure(figsize=(12, 3 * n))

    for i in range(n):
        row = sample_df.iloc[i]
        image_path = os.path.join(img_dir, row["image"])

        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        cropped, mask = segment_and_crop(image_rgb)

        plt.subplot(n, 3, 3 * i + 1)
        plt.imshow(image_rgb)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(n, 3, 3 * i + 2)
        plt.imshow(image_rgb)
        plt.imshow(mask, cmap="jet", alpha=0.3)
        plt.title("Mask overlay")
        plt.axis("off")

        plt.subplot(n, 3, 3 * i + 3)
        plt.imshow(cropped)
        plt.title(f"Cropped | Grade {row['myopic_maculopathy_grade']}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """
    Run the full baseline segmentation + classification pipeline.
    """
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # File paths
    train_csv = "Training/Training_LabelsDemographic.csv"
    test_csv = "Testing/Testing_LabelDemographic.csv"
    train_img_dir = "Training/Training_Images"
    test_img_dir = "Testing/Testing_Images"

    # Load labels
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nTraining class counts:")
    print(train_df["myopic_maculopathy_grade"].value_counts().sort_index())

    # Split training set into train and validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["myopic_maculopathy_grade"],
        random_state=42
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Show a few segmentation examples
    show_examples(train_df, train_img_dir, n=4)

    # Image transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Datasets
    train_dataset = MyopiaDataset(train_df, train_img_dir, transform=train_transform)
    val_dataset = MyopiaDataset(val_df, train_img_dir, transform=eval_transform)
    test_dataset = MyopiaDataset(test_df, test_img_dir, transform=eval_transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Model, loss and optimiser
    model = build_model(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_f1 = -1
    best_model_path = "baseline_segmentation_densenet121.pth"

    # Training loop
    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}/10")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model")

    # Load best checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Validation results
    print("\nValidation Results")
    val_loss, val_acc, val_f1, val_true, val_pred = evaluate(
        model, val_loader, criterion, device
    )
    print("Accuracy:", round(val_acc, 4))
    print("Macro F1:", round(val_f1, 4))
    print(classification_report(val_true, val_pred, digits=4))

    cm = confusion_matrix(val_true, val_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.show()

    # Test results
    print("\nTest Results")
    test_loss, test_acc, test_f1, test_true, test_pred = evaluate(
        model, test_loader, criterion, device
    )
    print("Accuracy:", round(test_acc, 4))
    print("Macro F1:", round(test_f1, 4))
    print(classification_report(test_true, test_pred, digits=4))


if __name__ == "__main__":
    main()
