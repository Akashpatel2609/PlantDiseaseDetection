# 🌿 Agri-Vision: Plant Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Smart Agriculture Solution**: An AI-powered deep learning system for automated detection of crop diseases and weeds using computer vision, enabling early intervention and sustainable crop management.

![Plant Disease Detection](https://img.shields.io/badge/Accuracy-96%2B%25-brightgreen)
![Disease Classes](https://img.shields.io/badge/Disease%20Classes-27-blue)
![Model Size](https://img.shields.io/badge/Model%20Size-37MB-lightblue)

## 🎯 Project Overview

Modern agriculture faces critical challenges in timely identifying crop diseases and invasive weeds, leading to significant yield loss. This project provides an automated solution using deep learning to detect multiple visual threats from field images.

### ✨ Key Features

- 🔍 **Real-time Disease Detection**: Upload plant images and get instant disease classification
- 🧠 **Advanced AI Model**: EfficientNetB0-based transfer learning with 96%+ accuracy
- 🌐 **Web Interface**: User-friendly Streamlit application for easy access
- ⚡ **GPU Accelerated**: Optimized for both CPU and GPU inference
- 📱 **27 Disease Classes**: Comprehensive coverage of major crop diseases
- 🚀 **Production Ready**: Complete training pipeline and deployment solution

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Image   │───▶│  Convolution NN │───▶│  Classification │
│   (224x224x3)  │    │                   │    │   Head (27)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────▼────────┐
                       │  Data Pipeline  │
                       │ • Normalization │
                       │ • Augmentation  │
                       │ • Preprocessing │
                       └─────────────────┘
```

## 🎯 Supported Disease Classes

| Plant | Diseases Detected |
|-------|------------------|
| **Apple** | Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Tomato** | Early Blight, Late Blight, Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Corn** | Common Rust, Northern Leaf Blight, Gray Leaf Spot, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| **Others** | Bell Pepper, Cherry, Orange, Peach, Strawberry, Raspberry, Soybean, Squash, Blueberry |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU support (optional but recommended)

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/PlantDiseaseDetection.git
cd PlantDiseaseDetection
```

### 2. Create Virtual Environment

```bash
python -m venv plantenv
# Windows
plantenv\Scripts\activate
# Linux/Mac
source plantenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirement.txt
```

### 4. Run the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 98.0% |
| **Validation Accuracy** | 96.1% |
| **Model Architecture** | EfficientNetB0 + Custom Head |
| **Parameters** | 9.7M (37MB) |
| **Training Strategy** | Two-Phase Transfer Learning |
| **Data Augmentation** | ✅ Rotation, Shift, Zoom, Flip |

### Training Results

- **Final Training Accuracy**: 98.0%
- **Final Validation Accuracy**: 96.1%
- **Training Loss**: 0.06 (final)
- **Validation Loss**: 0.14 (final)
- **Total Training Time**: ~2 hours (GPU) / ~8 hours (CPU)

## 🛠️ Project Structure

```
PlantDiseaseDetection/
├── main.py                          # Streamlit web application
├── Train_plant_disease.ipynb        # Model training notebook
├── Test_Plant_Disease.ipynb         # Model testing notebook
├── requirement.txt                  # Python dependencies
├── trained_model.keras              # Trained model file (37MB)
├── training_hist.json               # Training history & metrics
├── home_page.jpeg                   # App homepage image
├── Plant_Disease_Dataset/           # Training dataset
│   ├── train/                      # Training images (27 classes)
│   │   ├── Apple___Apple_scab/
│   │   ├── Tomato___Early_blight/
│   │   ├── Potato___healthy/
│   │   └── ...
│   ├── test/                       # Test images
│   └── valid/                      # Validation images
└── README.md                       # Project documentation
```

## 🔧 Usage

### Web Interface

1. **Launch Application**: Run `streamlit run main.py`
2. **Navigate Interface**: Use sidebar to switch between Home and Disease Detector
3. **Upload Image**: Click "Choose an Image" and select a plant photo
4. **Get Prediction**: Click "Predict" to see disease classification
5. **View Results**: See disease name and confidence score

### Programmatic Usage

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('trained_model.keras')

# Preprocess image
image = Image.open('plant_image.jpg').resize((224, 224))
image_array = np.array(image) / 255.0
image_batch = np.expand_dims(image_array, axis=0)

# Make prediction
prediction = model.predict(image_batch)
class_index = np.argmax(prediction)
confidence = np.max(prediction)

print(f"Predicted class: {class_index}")
print(f"Confidence: {confidence:.2%}")
```

## 🎓 Training Your Own Model

### Dataset Preparation

1. **Download Dataset**: [Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. **Organize Structure**:
   ```
   Plant_Disease_Dataset/
   ├── train/
   │   ├── class1/
   │   ├── class2/
   │   └── ... (27 classes total)
   └── test/
       ├── class1/
       ├── class2/
       └── ...
   ```

### Training Process

```bash
# Open Jupyter notebook
jupyter notebook Train_plant_disease.ipynb

# Follow the notebook cells step by step:
# 1. Data loading and preprocessing
# 2. Model architecture setup
# 3. Training with transfer learning
# 4. Model evaluation and saving
```

### Key Training Features

- **Transfer Learning**: Pre-trained EfficientNetB0 backbone
- **Data Augmentation**: Rotation, translation, zoom, brightness adjustment
- **Two-Phase Training**: Frozen → Fine-tuned approach
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **GPU Support**: Automatic GPU detection and utilization

## 🌐 Deployment Options

### Local Deployment
```bash
streamlit run main.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirement.txt
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.headless", "true"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Add `setup.sh` and `Procfile`
- **AWS/GCP**: Use container services

## 🔍 Model Details

### Architecture
- **Base Model**: EfficientNetB0 (ImageNet pre-trained)
- **Input Shape**: (224, 224, 3)
- **Classification Head**: 
  - Global Average Pooling
  - Dropout (0.3)
  - Dense(128, 'relu')
  - Dropout (0.5)
  - Dense(27, 'softmax')

### Training Configuration
- **Optimizer**: Adam (lr=0.001 → 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 30 (with early stopping)
- **Data Split**: 80% Train, 10% Validation, 10% Test

## 📈 Results & Evaluation

### Training Progress
The model achieved excellent performance with:
- **Final Training Accuracy**: 98.0%
- **Final Validation Accuracy**: 96.1%
- **Convergence**: Achieved in 10 epochs
- **Overfitting Control**: Minimal gap between training and validation

### Performance Metrics
| Phase | Training Acc | Validation Acc | Loss |
|-------|-------------|----------------|------|
| Initial | 59.3% | 83.7% | 1.39 |
| Final | 98.0% | 96.1% | 0.06 |

## 🛡️ Disease Detection Capabilities

The model can accurately detect and classify:

### Fruit Diseases
- **Apple**: Scab, Black Rot, Cedar Apple Rust
- **Grape**: Black Rot, Esca, Leaf Blight
- **Orange**: Huanglongbing (Citrus Greening)
- **Peach**: Bacterial Spot
- **Cherry**: Powdery Mildew

### Vegetable Diseases
- **Tomato**: Early/Late Blight, Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Yellow Leaf Curl Virus
- **Potato**: Early/Late Blight
- **Corn**: Common Rust, Northern Leaf Blight, Gray Leaf Spot
- **Bell Pepper**: Bacterial Spot
- **Squash**: Powdery Mildew

### Healthy Plant Detection
The model also identifies healthy plants across all supported species.

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add AmazingFeature'`
4. **Push to branch**: `git push origin feature/AmazingFeature`
5. **Open Pull Request**

### Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/PlantDiseaseDetection.git

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # or dev-env\Scripts\activate on Windows

# Install development dependencies
pip install -r requirement.txt
pip install jupyter matplotlib seaborn

# Start development
jupyter notebook
```

## 🙏 Acknowledgments

- **Dataset**: [Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Framework**: TensorFlow/Keras team for excellent deep learning tools
- **Model**: CNN for the revolutionary architecture
- **Community**: Streamlit team for the amazing web framework
- **Inspiration**: Agricultural research community working on sustainable farming

⭐ **Star this repository if it helped you!** ⭐

## 🚀 Future Enhancements

- [ ] Mobile app development (React Native/Flutter)
- [ ] Real-time video processing capabilities
- [ ] Disease severity assessment algorithms
- [ ] Treatment recommendations system
- [ ] Multi-language support interface
- [ ] Farmer dashboard with analytics
- [ ] Integration with IoT sensors
- [ ] Drone imagery support
- [ ] Crop yield prediction models
- [ ] Weather integration for disease risk assessment

## 🔬 Technical Specifications

### System Requirements
- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB+
- **Storage**: 2GB for model and dependencies
- **GPU**: Optional (NVIDIA with CUDA support)
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Performance Benchmarks
- **Inference Time**: ~200ms per image (CPU), ~50ms (GPU)
- **Model Size**: 37MB (optimized for deployment)
- **Memory Usage**: ~500MB during inference
- **Batch Processing**: Supports up to 32 images simultaneously

---
