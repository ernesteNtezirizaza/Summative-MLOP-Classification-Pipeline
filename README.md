# Brain Tumor MRI Classification - MLOPs Pipeline

## Project Description

This project implements an end-to-end Machine Learning Operations (MLOPs) pipeline for brain tumor classification from MRI images. The system classifies brain tumors into four categories: Glioma, Meningioma, No Tumor, and Pituitary tumor.

## Features

- **Image Classification**: Deep learning model for brain tumor classification
- **Feature Extraction**: Extracts image features and saves to CSV
- **Model Training & Evaluation**: Comprehensive evaluation with multiple metrics
- **REST API**: FastAPI-based prediction endpoints
- **Streamlit UI**: Interactive web interface with:
  - Single image prediction
  - Data visualizations (3+ feature interpretations)
  - Bulk data upload for retraining
  - Retraining trigger functionality
  - Model uptime monitoring
- **Retraining Pipeline**: Automated retraining with data upload
- **Load Testing**: Locust-based performance testing
- **Cloud Deployment**: Dockerized application ready for Render deployment

## Dataset

The dataset is from Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**Classes:**
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Project Structure

```
Summative-MLOP-Classification-Pipeline/
│
├── README.md
│
├── notebook/
│   └── brain_tumor_classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
│
├── data/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── test/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── models/
│   └── (model files will be saved here)
│
├── app.py (Streamlit UI)
├── api.py (FastAPI endpoints)
├── retrain.py (Retraining script)
├── locustfile.py (Load testing)
├── Dockerfile
├── requirements.txt
└── .dockerignore
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip
- Docker (optional, for containerization)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Summative-MLOP-Classification-Pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - Extract to the `data/` directory maintaining the train/test structure:
     ```
     data/
     ├── train/
     │   ├── glioma/
     │   ├── meningioma/
     │   ├── notumor/
     │   └── pituitary/
     └── test/
         ├── glioma/
         ├── meningioma/
         ├── notumor/
         └── pituitary/
     ```

5. **Extract features (Optional but Recommended)**
   ```bash
   python src/preprocessing.py
   ```
   This will create `data/processed/image_features_train.csv` and `data/processed/image_features_test.csv`.

6. **Train the model**
   
   Option A: Using Jupyter Notebook (Recommended)
   ```bash
   jupyter notebook notebook/brain_tumor_classification.ipynb
   ```
   
   Option B: Using Python Script
   ```bash
   python src/model.py
   ```

### Running the Application

#### Option 1: Streamlit UI (Recommended for local development)

```bash
streamlit run app.py
```

Access the UI at `http://localhost:8501`

#### Option 2: FastAPI Server

```bash
python api.py
```

API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Uptime: `http://localhost:8000/uptime`

#### Option 3: Docker

```bash
# Build the image
docker build -t brain-tumor-classifier .

# Run the container
docker run -p 8501:8501 brain-tumor-classifier
```

## Usage

### Prediction

1. **Via Streamlit UI:**
   - Navigate to the "Predict" tab
   - Upload a single MRI image
   - Click "Predict" to get the classification result

2. **Via API:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/image.jpg"
   ```

### Retraining

1. **Via Streamlit UI:**
   - Navigate to the "Retrain Model" tab
   - Upload multiple images (bulk upload)
   - Click "Trigger Retraining" button
   - Monitor the retraining process

2. **Via API:**
   ```bash
   curl -X POST "http://localhost:8000/retrain" \
        -F "files=@image1.jpg" \
        -F "files=@image2.jpg"
   ```

### Load Testing with Locust

```bash
# Terminal 1: Start API
python api.py

# Terminal 2: Run Locust
locust -f locustfile.py --host=http://localhost:8000

# Open browser: http://localhost:8089
```

## Testing the Pipeline

### Single Prediction

Test the prediction functionality directly:
```bash
python src/prediction.py data/test/glioma/image1.jpg
```

### Retraining

1. Upload images via Streamlit UI or API
2. Trigger retraining via UI or command line:
```bash
python retrain.py
```

## Model Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Loss**: Training and validation loss curves

## Deployment on Render

1. **Create a Render account** and connect your GitHub repository

2. **Create a new Web Service:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

3. **Environment Variables** (if needed):
   - Set any required environment variables in Render dashboard

4. **Deploy:**
   - Render will automatically deploy on push to main branch

## Video Demo

[YouTube Link - To be added]

## Results from Flood Request Simulation

Load testing results with different numbers of Docker containers will be documented here after running Locust tests.

## Troubleshooting

### Model Not Found
- Ensure you've trained the model first
- Check that `models/brain_tumor_model.h5` exists

### Import Errors
- Activate your virtual environment
- Install all requirements: `pip install -r requirements.txt`

### GPU Issues
- The code works on CPU, but GPU is recommended for training
- Install TensorFlow GPU version if you have CUDA

## Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **API Framework**: FastAPI
- **UI Framework**: Streamlit
- **Load Testing**: Locust
- **Containerization**: Docker
- **Cloud Platform**: Render

## File Structure Details

```
├── src/
│   ├── preprocessing.py  # Feature extraction
│   ├── model.py          # Model training
│   └── prediction.py     # Prediction functions
├── notebook/
│   └── brain_tumor_classification.ipynb  # Complete pipeline
├── app.py                # Streamlit UI
├── api.py                # FastAPI endpoints
├── retrain.py            # Retraining script
├── locustfile.py         # Load testing
├── Dockerfile            # Docker configuration
└── requirements.txt      # Dependencies
```

## Author

[Your Name]

## License

This project is for educational purposes.

