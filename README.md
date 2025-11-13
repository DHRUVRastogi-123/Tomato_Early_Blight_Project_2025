# Tomato Early Blight Detection 🍅

This project is a complete computer vision pipeline for detecting **Early Blight** disease in tomato plants. It uses a two-stage approach:

1. **Leaf Segmentation:** A **YOLOv8-Seg** model first identifies and segments all tomato leaves from the background.
2. **Disease Classification:** The segmented leaf images are then passed to an **EfficientNet-B0** classifier, which classifies them as "Healthy" or "Early Blight".

The project includes scripts for dataset preparation, model training, evaluation, and a Streamlit web application for real-time inference on images and videos.

---

## 🚀 Features

- **Two-Stage Pipeline:** Improves accuracy by first isolating the region of interest (the leaf) before classification.
- **Automated Dataset Curation:** Includes a script (`build_clean_classifier_dataset.py`) to automatically process a raw dataset of annotated images into a "clean" dataset of segmented, background-removed leaves for training the classifier.
- **Streamlit Web App:** A user-friendly interface (`streamlitapp.py`) to upload an image or video and receive real-time disease predictions.
- **Complete Workflow:** Provides all necessary scripts for training (`traincls.py`), evaluation (`test_cls.py`), and inference.
- **Cross-Platform Support:** Works seamlessly on Windows, macOS, and Linux.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.8 or higher** ([Download Python](https://www.python.org/downloads/))
- **Git** ([Download Git](https://git-scm.com/downloads))
- **CUDA 11.8+** (optional but recommended for GPU acceleration)

---

## 🔁 Workflow Overview

The project is structured into a clear, step-by-step workflow:

1. **Setup:** Install dependencies and configure the environment.
2. **Data Preparation:** Process raw annotated dataset into clean, segmented leaf images.
3. **Model Training:** Train the EfficientNet-B0 classifier on the prepared dataset.
4. **Model Evaluation:** Generate classification reports and confusion matrices.
5. **Run Application:** Launch the Streamlit app for real-time inference.

---

## 🛠️ Setup and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/tomato-disease-detection.git
cd tomato-disease-detection
```

Replace `YOUR_USERNAME` with your GitHub username.

---

### Step 2: Set Up the Environment

#### **Option A: Windows Users (Automatic Setup)**

We provide a batch script to automatically create and configure the virtual environment:

```bash
setup.bat
```

This script will:
- Create a virtual environment
- Activate it
- Install all dependencies from `requirements.txt`

#### **Option B: All Platforms (Manual Setup)**

Follow these steps for Windows, macOS, or Linux:

##### **Windows:**

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the environment
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

##### **macOS/Linux:**

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Verify Installation:**

```bash
python -c "import torch; print(torch.__version__)"
```

---

## 🏃 Running the Project

Follow these steps **in order** to prepare data, train the model, and run the application.

### Step 1: Prepare the Raw Dataset

Before running any scripts, you must provide your annotated dataset:

1. Create the following directory structure:
   ```
   Dataset/
   ├── Annotated/
   │   ├── train/
   │   ├── valid/
   │   └── test/
   ```

2. Place your **annotated images** and **YOLO-format label files** (.txt) in the respective folders.

3. Directory structure example:
   ```
   Dataset/Annotated/train/
   ├── image1.jpg
   ├── image1.txt
   ├── image2.jpg
   └── image2.txt
   ```

---

### Step 2: Build the Clean Classifier Dataset

This script uses the YOLOv8 segmenter to automatically segment leaves and create a clean training dataset.

#### **Windows:**
```bash
python scripts/build_clean_classifier_dataset.py
```

#### **macOS/Linux:**
```bash
python3 scripts/build_clean_classifier_dataset.py
```

**What it does:**
- Reads annotated images from `Dataset/Annotated/`
- Segments leaves using YOLOv8
- Saves clean, cropped leaf images to `data/processed/cls_clean_train/`, `data/processed/cls_clean_valid/`, and `data/processed/cls_clean_test/`

**Output:**
```
data/processed/
├── cls_clean_train/
│   ├── healthy/
│   └── early_blight/
├── cls_clean_valid/
│   ├── healthy/
│   └── early_blight/
└── cls_clean_test/
    ├── healthy/
    └── early_blight/
```

---

### Step 3: Train the Classifier

Train the EfficientNet-B0 model on the clean dataset.

#### **Windows:**
```bash
python scripts/traincls.py
```

#### **macOS/Linux:**
```bash
python3 scripts/traincls.py
```

**Training Configuration (in `traincls.py`):**
- Model: EfficientNet-B0
- Batch Size: 32
- Learning Rate: 0.0001
- Epochs: 25
- Optimizer: Adam

**Output:**
- Best model saved to: `models/best_cls_weights.pt`
- Training accuracy plot saved to: `models/accuracy_plot.png`

---

### Step 4: Evaluate the Model (Optional)

Generate detailed evaluation metrics and confusion matrix.

#### **Windows:**
```bash
python tests/test_cls.py
```

#### **macOS/Linux:**
```bash
python3 tests/test_cls.py
```

**Output:**
- Classification report: `reports/classification_report.txt`
- Confusion matrix plot: `reports/confusion_matrix.png`

---

### Step 5: Run the Streamlit Application

Launch the web application for real-time inference.

#### **Windows:**
```bash
streamlit run streamlitapp.py
```

#### **macOS/Linux:**
```bash
streamlit run streamlitapp.py
```

**Access the app:**
- Your browser will automatically open to: `http://localhost:8501`
- If not, manually navigate to this URL

**Features:**
- Upload single image or video file
- Real-time disease detection
- Annotated output with confidence scores
- Download processed video

---

## 📂 Project Structure

```
tomato-disease-detection/
├── Dataset/
│   └── Annotated/           # Raw annotated dataset
│       ├── train/
│       ├── valid/
│       └── test/
├── data/
│   └── processed/           # Clean segmented leaf crops
│       ├── cls_clean_train/
│       ├── cls_clean_valid/
│       └── cls_clean_test/
├── models/
│   ├── best_cls_weights.pt  # Trained classifier
│   └── accuracy_plot.png    # Training curves
├── reports/
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── scripts/
│   ├── traincls.py          # Training script
│   ├── build_clean_classifier_dataset.py  # Dataset preparation
│   ├── infer_model.py       # Inference logic
│   └── utils/
│       ├── segmenter.py     # YOLOv8 wrapper
│       ├── augmentations.py # Image augmentations
│       └── metrics.py       # Evaluation metrics
├── tests/
│   └── test_cls.py          # Evaluation script
├── streamlitapp.py          # Web application
├── requirements.txt         # Dependencies
├── setup.bat               # Windows setup script
└── README.md               # This file
```

---

## 📦 Key Files and Modules

| File | Purpose |
|------|---------|
| `streamlitapp.py` | Streamlit web application for inference |
| `requirements.txt` | Python dependencies |
| `setup.bat` | Automated setup for Windows users |
| `scripts/traincls.py` | Classification model training |
| `scripts/build_clean_classifier_dataset.py` | Dataset processing and preparation |
| `tests/test_cls.py` | Model evaluation and testing |
| `scripts/infer_model.py` | Core two-stage inference pipeline |
| `scripts/utils/segmenter.py` | YOLOv8 leaf segmentation wrapper |
| `scripts/utils/augmentations.py` | Data augmentation pipeline |
| `scripts/utils/metrics.py` | Evaluation metrics and reporting |
| `yolov8n-seg.pt` | Pre-trained YOLOv8 segmentation model |

---

## 🔧 Configuration and Customization

### Modifying Training Parameters

Edit `scripts/traincls.py` to adjust:

```python
NUM_EPOCHS = 25          # Number of training epochs
BATCH_SIZE = 32          # Batch size for training
LEARNING_RATE = 1e-4     # Optimizer learning rate
NUM_CLASSES = 2          # Number of disease classes
```

### Changing Model Architecture

To use a different backbone, modify:

```python
MODEL_NAME = "efficientnet_b0"  # Change to other timm models
```

Other available models: `efficientnet_b1`, `resnet50`, `vit_base_patch16_224`, etc.

---

## 🐛 Troubleshooting

### Issue: CUDA not found

**Solution:** Ensure NVIDIA drivers and CUDA toolkit are installed. Alternatively, CPU-only mode will work (slower):

```python
device = "cpu"  # Force CPU in scripts
```

### Issue: Out of memory errors

**Solution:** Reduce batch size in `traincls.py`:

```python
BATCH_SIZE = 16  # Reduce from 32
```

### Issue: Module not found errors

**Solution:** Verify virtual environment is activated and dependencies installed:

```bash
pip install -r requirements.txt --upgrade
```

### Issue: Streamlit app not opening

**Solution:** Try accessing manually at `http://localhost:8501` or restart:

```bash
streamlit run streamlitapp.py --logger.level=debug
```

---

## 📊 Expected Performance

Based on our evaluation on the test set:

- **Classification Accuracy:** 89-92%
- **Precision (Early Blight):** 0.97
- **Recall (Early Blight):** 0.91
- **F1-Score:** 0.94
- **Inference Time:** ~0.325 seconds per image

---

## 📚 Dependencies

Main libraries used in this project:

- **PyTorch:** Deep learning framework
- **Ultralytics YOLOv8:** Object detection and segmentation
- **timm:** PyTorch image models
- **Streamlit:** Web application framework
- **OpenCV:** Image processing
- **Albumentations:** Data augmentation
- **Scikit-learn:** Metrics and evaluation

See `requirements.txt` for complete list and versions.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgements

- **M.N.N.I.T. Allahabad** for academic support
- **Ultralytics** for YOLOv8
- **Hugging Face** for pre-trained models
- **PlantVillage** and **Kaggle** for datasets
- Open-source community contributors

---

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the project maintainers.

---

## 🌱 Future Enhancements

- [ ] Add mobile app deployment
- [ ] Support for multiple crop types
- [ ] Real-time drone integration
- [ ] Multi-disease detection
- [ ] Model compression for edge devices
- [ ] Advanced visualization dashboard

---

**Happy detecting! 🚀**
