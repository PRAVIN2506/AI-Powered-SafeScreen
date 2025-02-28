from flask import Flask, request, jsonify, send_from_directory
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import base64
import io

app = Flask(__name__)

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

def analyze_content(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/style.css')
def style():
    return send_from_directory('', 'style.css')

@app.route('/script.js')
def script():
    return send_from_directory('', 'script.js')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(',')[1]  # Remove the data URL scheme
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Analyze the content
    prediction, confidence = analyze_content(image)

    return jsonify({'prediction': prediction, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
