# 🛡️ AI-Powered SafeScreen with Content Redirection

SafeScreen is an AI-powered solution that monitors screen activity to protect children from inappropriate online content. It uses computer vision and deep learning (MobileNetV2) to classify screen captures in real time and seamlessly redirects children to safe digital environments when harmful content is detected. The project is built using Python, PyTorch, OpenCV, and automation tools to create a responsive and intelligent content moderation system.

---

## 📚 Table of Contents
- [📌 Features](#features)
- [📂 Folder Structure](#folder-structure)
- [🛠️ Setup Instructions](#setup-instructions)
- [🖼️ Dataset Preparation](#dataset-preparation)
- [🧠 Model Training](#model-training)
- [📊 Model Validation](#model-validation)
- [🖥️ Live Screen Monitoring](#live-screen-monitoring)
- [🔐 Redirection Logic](#redirection-logic)
- [💡 Future Enhancements](#future-enhancements)
- [📄 License](#license)

---

## 📌 Features

- ✅ Real-time screen capture every 3 seconds  
- 🧠 AI-based image classification using MobileNetV2  
- 🔐 Automatic content redirection to safe websites  
- 📊 Model accuracy reporting on validation data  
- 🎯 Binary classification: safe vs unsafe content  
- 🧪 Trainable with custom image datasets

---

## 📂 Folder Structure

```
📦 SafeScreen 
    ├── train_model.py 
    ├── validate_model.py 
    ├── capture_analyze_redirect.py
    ├── image dataset/ 
        │ ├── train/ 
        │ └── test/ 
    └── image_classification_model.pth
```

---

## 🛠️ Setup Instructions

**1.Install Python packages**:
   ```bash
   pip install torch torchvision tqdm pillow pyautogui
   ```
**2.Create your dataset folders**:

Place training and validation images in:
```
image dataset/
├── train/
│   ├── safe/
│   └── unsafe/
└── test/
    ├── safe/
    └── unsafe/

```

**3.Train the model**:

```bash
python train_model.py
```

**4.Validate the model**:

```bash
python validate_model.py
```

**5.Run live screen monitoring**:

```bash
python capture_analyze_redirect.py
```

## 🖼️ Dataset Preparation
- Ensure images are clear, labeled, and resized to 224x224 pixels
- Safe images go into safe/ folders; unsafe/adult content goes into unsafe/
- The labels used in training are:
```
 0 = safe
 1 = unsafe/adult
 ```

## 🧠 Model Training

- Architecture: MobileNetV2 (pretrained)
- Last Layer Modified: nn.Linear(..., 2) for binary classification
- Output Binary classifier (safe/unsafe)
- Epochs: 5
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Device: CUDA or CPU

The training script filters corrupted images and reports accuracy per epoch.

## 📊 Model Validation
- Uses validate_model.py
- Filters out corrupted images
- Prints overall validation accuracy
- Uses the same transformation pipeline for consistency

## 🖥️ Live Screen Monitoring
- capture_analyze_redirect.py captures screen every 3 seconds
- Classifies content using trained model
- If unsafe content is predicted with confidence > 0.9:
    - Closes current browser tab (Ctrl + W)
    - Redirects to: CoolMath Games

## 🔐 Redirection Logic
```python
if prediction == 1 and confidence > 0.9:
    pyautogui.hotkey('ctrl', 'w')
    webbrowser.open('https://www.coolmathgames.com/')
```

This ensures smooth redirection to a child-safe environment without abrupt blocking.

## 💡 Future Enhancements
- Expand dataset for multi-class classification (e.g. violence, substance abuse)
- Add voice alerts and activity logs
- Develop a GUI for parental dashboard customization
- Convert core logic into a browser extension

## 📄 License
This project is licensed under the MIT License. Feel free to use, modify, and distribute with attribution.
