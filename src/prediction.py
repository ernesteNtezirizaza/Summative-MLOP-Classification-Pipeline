"""
Prediction module for brain tumor classification.
Handles single image predictions and batch predictions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import pickle
import os


class BrainTumorPredictor:
    """
    Predictor class for brain tumor classification.
    """
    
    def __init__(self, model_path='models/brain_tumor_model.h5', class_names_path='models/class_names.pkl'):
        """
        Initialize the predictor with a trained model.
        """
        self.model = None
        self.class_names = None
        self.img_size = (224, 224)
        
        if os.path.exists(model_path):
            self.load_model(model_path, class_names_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def load_model(self, model_path='models/brain_tumor_model.h5', class_names_path='models/class_names.pkl'):
        """
        Load the trained model and class names.
        """
        self.model = keras.models.load_model(model_path)
        
        # Define valid brain tumor classes
        valid_classes = {'glioma', 'meningioma', 'notumor', 'pituitary'}
        
        # Store original class names from pickle for mapping model outputs
        self.original_class_names = None
        self.class_mapping = None  # Maps original indices to filtered indices
        
        if os.path.exists(class_names_path):
            with open(class_names_path, 'rb') as f:
                loaded_class_names = pickle.load(f)
            
            # Store original class names for mapping
            self.original_class_names = loaded_class_names
            
            # Filter out 'unknown' and any other invalid classes
            self.class_names = [name for name in loaded_class_names if name in valid_classes]
            
            # Create mapping from original class indices to filtered class indices
            self.class_mapping = {}
            filtered_idx = 0
            for orig_idx, class_name in enumerate(loaded_class_names):
                if class_name in valid_classes:
                    self.class_mapping[orig_idx] = filtered_idx
                    filtered_idx += 1
            
            # If filtering removed classes, warn and use defaults
            if len(self.class_names) != len(loaded_class_names):
                removed = [name for name in loaded_class_names if name not in valid_classes]
                print(f"Warning: Removed invalid classes from model: {removed}")
                print(f"Using valid classes only: {self.class_names}")
            
            # Ensure we have valid classes
            if not self.class_names:
                print("Warning: No valid classes found in class_names.pkl. Using defaults.")
                self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
                self.class_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        else:
            # Default class names if file doesn't exist
            self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
            self.class_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        
        print(f"Model loaded successfully. Classes: {self.class_names}")
    
    def preprocess_image(self, image_path_or_array):
        """
        Preprocess a single image for prediction.
        """
        # Load image
        if isinstance(image_path_or_array, str):
            # Load from file path
            img = cv2.imread(image_path_or_array)
            if img is None:
                raise ValueError(f"Could not load image from {image_path_or_array}")
        else:
            # Assume it's already a numpy array
            img = image_path_or_array
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, self.img_size)
        
        # Normalize pixel values to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path_or_array, return_probabilities=False):
        """
        Predict the class of a single image.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            return_probabilities: If True, return all class probabilities
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_img = self.preprocess_image(image_path_or_array)
        
        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        
        # Get the original model's predicted class index
        original_predicted_idx = np.argmax(predictions[0])
        
        # Map to filtered class index if model has 'unknown' or other invalid classes
        if self.class_mapping and original_predicted_idx in self.class_mapping:
            # Valid class, use mapped index
            predicted_class_idx = self.class_mapping[original_predicted_idx]
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][original_predicted_idx])
        elif original_predicted_idx < len(self.class_names):
            # Direct mapping (no filtering needed)
            predicted_class_idx = original_predicted_idx
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][original_predicted_idx])
        else:
            # Invalid prediction (e.g., 'unknown' class), find best valid class
            # Get probabilities for valid classes only
            valid_probs = []
            valid_indices = []
            for orig_idx, class_name in enumerate(self.original_class_names if self.original_class_names else self.class_names):
                if class_name in {'glioma', 'meningioma', 'notumor', 'pituitary'}:
                    valid_probs.append(predictions[0][orig_idx])
                    valid_indices.append(orig_idx)
            
            if valid_probs:
                best_valid_idx = np.argmax(valid_probs)
                original_best_idx = valid_indices[best_valid_idx]
                predicted_class_idx = self.class_mapping.get(original_best_idx, 0)
                predicted_class = self.class_names[predicted_class_idx]
                confidence = float(predictions[0][original_best_idx])
            else:
                # Fallback
                predicted_class_idx = 0
                predicted_class = self.class_names[0]
                confidence = float(predictions[0][0])
        
        # Prepare result
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_index': int(predicted_class_idx)
        }
        
        # Add all probabilities if requested (only for valid classes)
        if return_probabilities:
            probabilities = {}
            if self.original_class_names and self.class_mapping:
                # Map from original model outputs to filtered class names
                for orig_idx, class_name in enumerate(self.original_class_names):
                    if class_name in {'glioma', 'meningioma', 'notumor', 'pituitary'}:
                        if orig_idx < len(predictions[0]):
                            filtered_idx = self.class_mapping.get(orig_idx)
                            if filtered_idx is not None:
                                probabilities[self.class_names[filtered_idx]] = float(predictions[0][orig_idx])
            else:
                # Direct mapping
                for idx, class_name in enumerate(self.class_names):
                    if idx < len(predictions[0]):
                        probabilities[class_name] = float(predictions[0][idx])
            
            result['probabilities'] = probabilities
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image file paths
            return_probabilities: If True, return all class probabilities
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for img_path in image_paths:
            try:
                result = self.predict(img_path, return_probabilities)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results


def load_predictor(model_path='models/brain_tumor_model.h5'):
    """
    Convenience function to load a predictor instance.
    """
    return BrainTumorPredictor(model_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prediction.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load predictor
    predictor = load_predictor()
    
    # Make prediction
    result = predictor.predict(image_path, return_probabilities=True)
    
    print(f"\nPrediction for {image_path}:")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nAll Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")

