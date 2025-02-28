import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from PIL import Image

# Define image size and batch size
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = 'C:/Users/BEBE MINE/Downloads/image dataset'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom pil_loader function to handle corrupted images
def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except OSError:
        print(f"Skipping corrupted image: {path}")
        return None

# Custom ImageFolder class to use the custom pil_loader
class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        if sample is None:
            return None, None
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

# Load datasets with the CustomImageFolder
train_dataset = CustomImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = CustomImageFolder(root=os.path.join(DATA_DIR, 'test'), transform=transform)

# Filter out corrupted images
train_dataset.samples = [(s, t) for s, t in train_dataset.samples if pil_loader(s) is not None]
val_dataset.samples = [(s, t) for s, t in val_dataset.samples if pil_loader(s) is not None]

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the pre-trained model
model = models.mobilenet_v2(pretrained=True)

# Modify the classifier to match the number of classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Assuming binary classification

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):  # Reduced to 5 epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    torch.save(model.state_dict(), 'image_classification_model.pth')
    print("Model saved")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)  # Reduced to 5 epochs
