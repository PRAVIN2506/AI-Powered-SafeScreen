import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import torchvision.models as models

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

# Load validation dataset with the CustomImageFolder
val_dataset = CustomImageFolder(root=os.path.join(DATA_DIR, 'test'), transform=transform)

# Filter out corrupted images
val_dataset.samples = [(s, t) for s, t in val_dataset.samples if pil_loader(s) is not None]

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Recreate the model architecture
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # Assuming binary classification

# Load the saved model weights
model.load_state_dict(torch.load('image_classification_model.pth'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        # Skip corrupted images
        if images is None or labels is None:
            continue
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = 100 * correct / total
print(f'Validation Accuracy: {val_acc:.2f}%')
