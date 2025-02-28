import pyautogui
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import webbrowser
import time
import os

# Define image size
IMG_SIZE = 224

# Define transformation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load your trained model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load('image_classification_model.pth'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def analyze_content(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

def capture_and_analyze():
    frame_count = 0
    while True:
        # Capture screen
        frame_count += 1
        screenshot = pyautogui.screenshot()
        filename = f'frame_{frame_count}.png'
        screenshot.save(filename)
        print(f'Captured {filename}')
        
        # Analyze content
        prediction, confidence = analyze_content(filename)
        print(f'Prediction for {filename}: {prediction} with confidence {confidence:.2f}')
        
        # Delete the image after analysis
        os.remove(filename)
        
        # Redirect if adult content is detected with high confidence
        if prediction == 1 and confidence > 0.9:  # Assuming 1 indicates adult content and confidence > 0.9
            print("Redirecting to a safe page...")
            # Close the current browser tab
            pyautogui.hotkey('ctrl', 'w')
            # Open the kids' game
            webbrowser.open('https://www.coolmathgames.com/')
            break
        
        time.sleep(3)  # Capture every 3 seconds

if __name__ == "__main__":
    capture_and_analyze()
