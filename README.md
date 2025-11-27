# Brain Tumor MRI Classification - MLOPs Pipeline

## 🔗 Quick Links

- **Video Demo**: [https://www.youtube.com/watch?v=UpqlYGE9rAc](https://www.youtube.com/watch?v=UpqlYGE9rAc)
- **App Base URL**: [https://brain-tumor-classifier-xcon.onrender.com/](https://brain-tumor-classifier-xcon.onrender.com/)
- **App Swagger Doc URL**: [https://brain-tumor-classifier-xcon.onrender.com/docs/](https://brain-tumor-classifier-xcon.onrender.com/docs/)

## Project Description

This project implements an end-to-end Machine Learning Operations (MLOPs) pipeline for brain tumor classification from MRI images. The system classifies brain tumors into four categories: Glioma, Meningioma, No Tumor, and Pituitary tumor.

## Features

- **Image Classification**: Deep learning model for brain tumor classification
- **Feature Extraction**: Extracts image features and saves to CSV
- **Model Training & Evaluation**: Comprehensive evaluation with multiple metrics
- **Database System**: SQLite database for tracking:
  - Uploaded images and metadata
  - Preprocessing activities and logs
  - Training sessions with complete metrics history
- **REST API**: FastAPI-based prediction endpoints
- **Web Interface**: Modern HTML/CSS/JavaScript UI with:
  - Single image prediction
  - Data visualizations (3+ feature interpretations)
  - Bulk data upload for retraining (saved to database)
  - Retraining trigger functionality
  - Model uptime monitoring
  - Database statistics and training history
- **Retraining Pipeline**: Automated retraining with data upload and database tracking
- **Load Testing**: Locust-based performance testing
- **Cloud Deployment**: Dockerized application ready for deployment on Render

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
├── README.md                  # Project documentation
├── Dockerfile                 # Docker container configuration
├── requirements.txt           # Python dependencies
├── .dockerignore              # Docker ignore patterns
│
├── api.py                     # FastAPI REST endpoints and web server
├── retrain.py                 # Retraining script with database logging
├── locustfile.py              # Load testing configuration
│
├── notebook/
│   └── brain_tumor_classification.ipynb  # Complete ML pipeline notebook
│
├── src/
│   ├── preprocessing.py       # Feature extraction from images
│   ├── model.py               # Model training and evaluation
│   ├── prediction.py          # Prediction functions
│   └── database.py            # Database management and tracking
│
├── static/
│   ├── index.html             # Main web interface (HTML)
│   ├── app.js                 # Frontend JavaScript
│   └── style.css              # Styling (Poppins font family)
│
├── data/
│   ├── train/                 # Training images organized by class
│   │   ├── glioma/            # Glioma tumor images
│   │   ├── meningioma/        # Meningioma tumor images
│   │   ├── notumor/           # No tumor images
│   │   └── pituitary/         # Pituitary tumor images
│   ├── test/                  # Test images organized by class
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   ├── processed/             # Processed feature data
│   │   ├── image_features_train.csv
│   │   ├── image_features_test.csv
│   │   └── image_features_retrain_temp.csv  # Temporary retraining features
│   ├── retrain_uploads/       # Uploaded images for retraining (flat directory)
│   │                           # Files saved with class prefix: {class_name}_{filename}
│   └── retraining_database.db # SQLite database for tracking uploads and training
│
└── models/
    ├── brain_tumor_model.h5           # Original trained model weights
    ├── brain_tumor_model_retrained.h5 # Retrained model weights
    ├── class_names.pkl                # Class label mappings
    ├── models/                        # Nested models directory
    │   └── visualizations/            # Retraining visualizations
    │       ├── confusion_matrix_retrain.png
    │       └── training_history_retrain.png
    └── visualizations/                # Model training visualizations
        ├── class_distribution.png
        ├── sample_images.png
        ├── feature_distributions.png
        ├── training_history.png
        ├── learning_curves.png
        ├── confusion_matrix_validation.png
        ├── confusion_matrix_validation_notebook.png
        ├── confusion_matrix_test.png
        └── sample_prediction.png
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip
- Docker (optional, for containerization)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ernesteNtezirizaza/Summative-MLOP-Classification-Pipeline
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

## Running the Application

### Option 1: FastAPI Server (Recommended)

```bash
python api.py
```

API will be available at `http://localhost:8000`
- **Web Interface**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Uptime**: `http://localhost:8000/uptime`

### Option 2: Streamlit UI (Alternative)

```bash
streamlit run app.py
```

Access the UI at `http://localhost:8501`

### Option 3: Docker

```bash
# Build the image
docker build -t brain-tumor-classifier .

# Run the container
docker run -p 8000:8000 -e PORT=8000 brain-tumor-classifier
```

## Usage

### Prediction

1. **Via Web Interface:**
   - Navigate to `http://localhost:8000`
   - Click on "Predict" in the sidebar
   - Upload a single MRI image
   - Click "Predict" to get the classification result

2. **Via API:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/image.jpg"
   ```

### Retraining

**Important:** The retraining process uses **only newly uploaded data** (`data/retrain_uploads/`) for fine-tuning. The existing model is loaded as a pre-trained model and fine-tuned on the new data only.

1. **Via Web Interface:**
   - Navigate to the "Upload Data" tab
   - Upload multiple images (bulk upload) - **automatically saved to database**
   - Navigate to "Retrain Model" tab
   - View database statistics and recent training sessions
   - Adjust training epochs (default: 3) and fine-tuning epochs (default: 1)
   - Click "Trigger Retraining" button
   - Monitor the retraining process (all activities logged to database)
   - **Note:** Retraining uses only newly uploaded data (original model preserved)

2. **Via API:**
   ```bash
   # Upload images (saved to database)
   curl -X POST "http://localhost:8000/retrain" \
        -F "files=@image1.jpg" \
        -F "files=@image2.jpg" \
        -F "class_name=glioma"
   
   # Trigger retraining
   curl -X POST "http://localhost:8000/retrain/trigger" \
        -H "Content-Type: application/json" \
        -d '{"epochs": 3, "fine_tune_epochs": 1}'
   
   # Check training status
   curl "http://localhost:8000/retrain/status"
   
   # Get database statistics
   curl "http://localhost:8000/database/stats"
   ```

3. **Via Command Line:**
   ```bash
   python retrain.py [epochs] [fine_tune_epochs]
   ```

### Load Testing with Locust

```bash
# Terminal 1: Start API
python api.py

# Terminal 2: Run Locust
locust -f locustfile.py --host=http://localhost:8000

# Open browser: http://localhost:8089
```

## Database Implementation

The project includes a comprehensive SQLite database system that tracks the complete retraining pipeline:

### Database Features

1. **Data File Uploading + Saving to Database**
   - All uploaded images are automatically saved to the database
   - Tracks: filename, class, file path, size, upload timestamp, metadata
   - Accessible via Web Interface and FastAPI endpoints

2. **Data Preprocessing Logging**
   - All preprocessing activities are logged to the database
   - Tracks: images processed, features extracted, processing time, status
   - Images are marked as "processed" after preprocessing

3. **Retraining Session Tracking**
   - Complete training history with metrics
   - Tracks: epochs, accuracy, precision, recall, F1-score (before/after)
   - Links uploaded images to training sessions
   - Full audit trail for compliance

### Database Schema

- **`uploaded_images`**: Stores metadata for all uploaded images
- **`training_sessions`**: Tracks each retraining session with metrics
- **`preprocessing_logs`**: Logs preprocessing activities and timing

### Database Access

The database is automatically created at `data/retraining_database.db` on first use.

**View Statistics:**
- Web Interface: "Retrain Model" page shows database statistics
- API: `GET /database/stats` endpoint
- Python: `from src.database import get_database`

## Model Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Loss**: Training and validation loss curves

All metrics are automatically saved to the database for each training session.

## Deployment

### Deployment on Render

Render is an excellent cloud platform for deploying Docker applications.

#### Quick Start (5 Minutes)

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Create Render account** at https://render.com and connect GitHub

3. **Create new Web Service:**
   - Click "New +" → "Web Service"
   - Select your repository
   - **IMPORTANT**: Choose **"Docker"** as build method (not Nixpacks)
   - Render will automatically detect your Dockerfile

4. **Configure:**
   - **Name**: `brain-tumor-classifier` (or your choice)
   - **Region**: Choose closest to users
   - **Branch**: `main`
   - **Instance Type**: Starter ($7/month) recommended (Free tier may not work with TensorFlow)

5. **Environment Variables** (optional):
   - `PORT`: Automatically set by Render
   - `PYTHONUNBUFFERED`: `1` (recommended)

6. **Deploy:**
   - Click "Create Web Service"
   - Wait for build (5-15 minutes)
   - Your app will be live at: `https://your-app-name.onrender.com`

#### Render Important Notes

- **Dockerfile**: Must use `${PORT}` environment variable (already configured)
- **Model File**: Ensure `models/brain_tumor_model.h5` exists in repository
- **Free Tier**: 512 MB RAM may be insufficient for TensorFlow
- **Recommended**: Use Starter plan ($7/month) or higher for better performance

### Docker Configuration

The project includes a Dockerfile configured for cloud deployment:

- **Base Image**: Python 3.9-slim
- **Port**: Uses `PORT` environment variable (defaults to 8000)
- **Health Check**: Configured for `/health` endpoint
- **FastAPI Server**: Runs with uvicorn

#### Local Docker Testing

```bash
# Build Docker image
docker build -t brain-tumor-app .

# Run Docker container
docker run -p 8000:8000 -e PORT=8000 brain-tumor-app

# Test in browser
# Open http://localhost:8000
# Open http://localhost:8000/docs for API docs
```

### Deployment Checklist

Before deploying, ensure:

- [ ] Dockerfile exists and uses `${PORT}` environment variable
- [ ] All code is committed to Git
- [ ] Code is pushed to GitHub
- [ ] Model file exists (`models/brain_tumor_model.h5`)
- [ ] requirements.txt is complete
- [ ] .dockerignore is configured
- [ ] Account created on Render
- [ ] GitHub connected to deployment platform
- [ ] Deployment successful
- [ ] Application tested and working

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

### Retraining Errors
- Check that uploaded images exist in `data/retrain_uploads/`
- Verify database is accessible: `data/retraining_database.db`
- Check server logs for detailed error messages
- Ensure sufficient memory for TensorFlow operations

### Deployment Issues

**Build Fails:**
- Check Dockerfile syntax
- Verify all files are in repository
- Check requirements.txt has all dependencies

**App Crashes on Startup:**
- Check application logs
- Verify model file exists
- Check file paths are correct
- Verify PORT environment variable is used correctly

**Out of Memory:**
- Upgrade to higher plan (Starter or Standard)
- Optimize model loading (lazy loading)
- Reduce batch sizes

## Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Database**: SQLite (for tracking and audit trail)
- **API Framework**: FastAPI
- **UI Framework**: HTML/CSS/JavaScript (Poppins font)
- **Alternative UI**: Streamlit
- **Load Testing**: Locust
- **Containerization**: Docker
- **Cloud Platform**: Render
- **Data Visualization**: Matplotlib, Seaborn, Plotly

## API Endpoints

### Main Endpoints

- `GET /` - Web interface (HTML)
- `GET /api` - API information
- `GET /health` - Health check endpoint
- `GET /uptime` - Uptime and statistics
- `GET /docs` - Interactive API documentation (Swagger)
- `GET /redoc` - Alternative API documentation

### Prediction Endpoints

- `POST /predict` - Predict brain tumor class from single image
- `POST /predict/batch` - Batch prediction for multiple images

### Retraining Endpoints

- `POST /retrain` - Upload images for retraining
- `POST /retrain/trigger` - Trigger model retraining
- `GET /retrain/status` - Get retraining status
- `GET /retrain/sessions` - Get recent training sessions

### Database Endpoints

- `GET /database/stats` - Get database statistics

### Visualization Endpoints

- `GET /visualizations/data` - Get feature data for visualizations

## Author

Erneste Ntezirizaza

## License

This project is for educational purposes.
