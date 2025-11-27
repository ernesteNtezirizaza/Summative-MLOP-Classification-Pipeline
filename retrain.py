"""
Retraining script for brain tumor classification model.
Handles data upload, preprocessing, and model retraining.
"""

import os
import shutil
import numpy as np
from datetime import datetime
import time
import tensorflow as tf
try:
    from src.model import BrainTumorClassifier
    from src.preprocessing import extract_features_from_directory, prepare_data_for_training
    from src.database import get_database
except ImportError:
    import sys
    sys.path.insert(0, 'src')
    from model import BrainTumorClassifier
    from preprocessing import extract_features_from_directory, prepare_data_for_training
    from database import get_database
import pickle


# NOTE: The retraining flow uses prepare_retrain_data_from_database() which:
# 1. Reads from data/retrain_uploads (where files are uploaded)
# 2. Creates a temporary directory data/temp_retrain for training
# 3. Validates class names (only glioma, meningioma, notumor, pituitary)
# 4. Does NOT move files to data/train/unknown or any other permanent location
# This ensures retraining uses only the retrain_uploads folder, not data/train/unknown


def prepare_retrain_data_from_database(db, temp_data_dir='data/temp_retrain', retrain_data_dir='data/retrain_uploads', main_data_dir='data/train'):
    """
    Prepare retraining data from database (newly uploaded, unprocessed images only).
    Creates a temporary directory with only new data for retraining.
    
    IMPORTANT: This function uses ONLY NEWLY UPLOADED data (from retrain_uploads).
    The existing model is loaded as a pre-trained model and fine-tuned on the new data only.
    
    Files are copied to a temporary directory for training. Original files are preserved.
    
    Args:
        db: Database instance
        temp_data_dir: Temporary directory to store new data (default: data/temp_retrain)
        retrain_data_dir: Directory where retraining uploads are stored (default: data/retrain_uploads)
        main_data_dir: Directory containing original training data (unused, kept for compatibility)
    
    Returns:
        tuple: (temp_data_dir, image_count, image_ids) or (None, 0, []) if no data
    """
    print(f"Preparing retraining data from database (newly uploaded images only)...")
    
    uploaded_images = db.get_uploaded_images(processed=False)
    
    if not uploaded_images:
        print("No unprocessed images found in database")
        # Fallback: If no database images, check retrain_uploads directory
        if os.path.exists(retrain_data_dir):
            print("No database images found, checking retrain_uploads directory...")
            # Check if there are any files in retrain_uploads
            has_files = False
            for root, dirs, files in os.walk(retrain_data_dir):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    has_files = True
                    break
            if not has_files:
                return None, 0, []
        else:
            return None, 0, []
    
    # Clean up old temp directory if it exists
    if os.path.exists(temp_data_dir):
        shutil.rmtree(temp_data_dir)
    os.makedirs(temp_data_dir, exist_ok=True)
    
    image_count = 0
    image_ids = []
    
    # Define valid brain tumor classes
    valid_classes = {'glioma', 'meningioma', 'notumor', 'pituitary'}
    
    print(f"Copying newly uploaded data from database...")
    for img_record in uploaded_images:
        file_path = img_record.get('file_path')
        class_name = img_record.get('class_name', 'glioma')  # Default to 'glioma' if not specified
        image_id = img_record.get('id')
        
        # Skip images with invalid class names
        if class_name not in valid_classes:
            print(f"Warning: Skipping image ID {image_id} with invalid class name '{class_name}'. Valid classes are: {', '.join(valid_classes)}")
            continue
        
        if not file_path or not os.path.exists(file_path):
            print(f"Warning: File not found for image ID {image_id}: {file_path}")
            continue
        
        # Create class directory in temp folder
        class_dir = os.path.join(temp_data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy file to temp directory
        filename = os.path.basename(file_path)
        target_path = os.path.join(class_dir, filename)
        
        # Handle filename conflicts by adding a prefix
        if os.path.exists(target_path):
            base, ext = os.path.splitext(filename)
            target_path = os.path.join(class_dir, f"new_{base}_{image_id}{ext}")
        
        try:
            shutil.copy2(file_path, target_path)
            image_count += 1
            image_ids.append(image_id)
        except Exception as e:
            print(f"Error copying {file_path}: {str(e)}")
    
    print(f"Prepared {image_count} newly uploaded images for retraining in {temp_data_dir}")
    return temp_data_dir, image_count, image_ids


def retrain_model(retrain_data_dir='data/retrain_uploads', 
                  main_data_dir='data/train',
                  epochs=10,
                  fine_tune_epochs=3,
                  model_save_path='models/brain_tumor_model_retrained.h5'):
    """
    Main retraining function.
    
    IMPORTANT: This function uses ONLY NEWLY UPLOADED data (from retrain_uploads) 
    for retraining. The existing model is loaded as a pre-trained model and 
    fine-tuned on the new data only.
    
    Steps:
    1. Create training session in database
    2. Prepare retraining data from database (newly uploaded images only)
    3. Extract features from new data (with database logging)
    4. Prepare data generators from newly uploaded data only
    5. Load existing model (as pre-trained model)
    6. Retrain/fine-tune the model on new data only
    7. Evaluate and save
    8. Update database with results
    9. Clean up temporary directories
    """
    # Initialize database
    db = get_database()
    
    # Create training session in database
    training_session_id = db.create_training_session(
        epochs=epochs,
        fine_tune_epochs=fine_tune_epochs,
        model_path=model_save_path,  # This will be brain_tumor_model_retrained.h5
        notes=f"Retraining with newly uploaded data only ({retrain_data_dir}). Original model preserved."
    )
    
    print("=" * 50)
    print("Starting Model Retraining Process")
    print(f"Training Session ID: {training_session_id}")
    print("=" * 50)
    
    # Update status to in_progress
    db.update_training_session(training_session_id, status='in_progress')
    
    # Step 1: Prepare retraining data from database (newly uploaded images only)
    print("\nStep 1: Preparing retraining data from database (newly uploaded images only)...")
    temp_data_dir, image_count, image_ids = prepare_retrain_data_from_database(
        db, 
        retrain_data_dir=retrain_data_dir,
        main_data_dir=main_data_dir
    )
    
    if not temp_data_dir or image_count == 0:
        print("No new data to retrain with. Exiting.")
        db.update_training_session(training_session_id, status='failed', 
                                   notes="No new data to retrain with")
        return False
    
    print(f"Prepared {image_count} newly uploaded images for retraining")
    
    # Step 2: Extract features from new data only
    print("\nStep 2: Extracting features from newly uploaded data...")
    preprocessing_start = time.time()
    images_processed = image_count
    features_extracted = 0
    
    try:
        # Extract features from temporary directory (new data only)
        temp_features_csv = 'data/processed/image_features_retrain_temp.csv'
        features_df = extract_features_from_directory(temp_data_dir, temp_features_csv)
        features_extracted = len(features_df) if features_df is not None else 0
        
        preprocessing_time = time.time() - preprocessing_start
        
        # Log preprocessing to database
        db.log_preprocessing(
            training_session_id=training_session_id,
            images_processed=images_processed,
            features_extracted=features_extracted,
            processing_time=preprocessing_time,
            status='completed'
        )
        
        # Mark images as processed
        if image_ids:
            db.mark_images_processed(image_ids, training_session_id)
        
        print(f"Feature extraction completed: {images_processed} images, {features_extracted} features")
    except Exception as e:
        preprocessing_time = time.time() - preprocessing_start
        error_msg = str(e)
        print(f"Warning: Feature extraction failed: {error_msg}")
        print("Continuing with training...")
        
        # Log preprocessing failure
        db.log_preprocessing(
            training_session_id=training_session_id,
            images_processed=images_processed,
            features_extracted=0,
            processing_time=preprocessing_time,
            status='failed',
            error_message=error_msg
        )
    
    # Step 3: Prepare data generators (using only newly uploaded data)
    print("\nStep 3: Preparing data generators (newly uploaded data only)...")
    try:
        # Require at least 2 images for retraining (need at least 1 for validation)
        if image_count < 2:
            error_msg = f"Insufficient images for retraining. Found {image_count} image(s), need at least 2 (1 for training, 1 for validation)."
            print(f"Error: {error_msg}")
            # Clean up temp directory
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir)
            db.update_training_session(training_session_id, status='failed', 
                                       notes=error_msg)
            return False
        
        # Use smaller batch size for Render's memory constraints (512MB RAM)
        # Reduced from default 32 to 4 for small datasets
        batch_size = min(4, max(1, image_count // 10))  # Adaptive batch size based on data size
        train_gen, val_gen = prepare_data_for_training(temp_data_dir, batch_size=batch_size)
        class_names = list(train_gen.class_indices.keys())
        
        # Define valid brain tumor classes
        valid_classes = {'glioma', 'meningioma', 'notumor', 'pituitary'}
        
        # Filter out invalid classes (unknown, retraining_uploads, retrain_uploads, etc.)
        invalid_classes = [name for name in class_names if name not in valid_classes]
        if invalid_classes:
            print(f"Warning: Invalid class names found in data: {invalid_classes}")
            print(f"These classes will be filtered out. Only valid classes will be used for training.")
            class_names = [name for name in class_names if name in valid_classes]
            print(f"Filtered classes: {class_names}")
        
        # Ensure we have at least one valid class
        if not class_names:
            error_msg = "No valid brain tumor classes found. Valid classes are: glioma, meningioma, notumor, pituitary."
            print(f"Error: {error_msg}")
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir)
            db.update_training_session(training_session_id, status='failed', 
                                       notes=error_msg)
            return False
        
        num_classes = len(class_names)
        print(f"Found {num_classes} classes: {class_names}")
        print(f"Using batch size: {batch_size} (optimized for small dataset and Render's memory constraints)")
        print(f"Training samples: {train_gen.samples}, Validation samples: {val_gen.samples if val_gen else 0}")
        
        # Warn about small datasets
        if image_count < 20:
            print(f"\n⚠️  WARNING: Retraining with only {image_count} newly uploaded images ({train_gen.samples} training, {val_gen.samples} validation)")
            print(f"   Very small datasets can lead to:")
            print(f"   - Overfitting (model memorizes the data)")
            print(f"   - Unrealistic 100% accuracy (not generalizable)")
            print(f"   - Poor performance on new, unseen images")
            print(f"   Recommendation: Upload at least 20-30 images per class for reliable results.")
        
        # Validate we have enough data
        if train_gen.samples < num_classes:
            error_msg = f"Insufficient training samples ({train_gen.samples}) for {num_classes} classes. Need at least {num_classes} samples."
            print(f"Error: {error_msg}")
            # Clean up temp directory
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir)
            db.update_training_session(training_session_id, status='failed', 
                                       notes=error_msg)
            return False
        
        # Check validation generator
        if val_gen is None or val_gen.samples == 0:
            error_msg = "No validation samples available. Cannot train without validation data. Please upload at least 2 images."
            print(f"Error: {error_msg}")
            # Clean up temp directory
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir)
            db.update_training_session(training_session_id, status='failed', 
                                       notes=error_msg)
            return False
        
        print(f"Data validation passed: {train_gen.samples} training samples, {val_gen.samples} validation samples")
        
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up temp directory
        if os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir)
        db.update_training_session(training_session_id, status='failed', 
                                   notes=f"Error preparing data: {str(e)}")
        return False
    
    # Step 4: Load or create model
    print("\nStep 4: Loading existing model as pre-trained model...")
    # Determine models directory (project root)
    current_dir = os.path.abspath(os.getcwd())
    if os.path.basename(current_dir) == 'notebook':
        models_dir = os.path.join(os.path.dirname(current_dir), 'models')
    elif os.path.exists(os.path.join(current_dir, 'src')):
        models_dir = os.path.join(current_dir, 'models')
    else:
        models_dir = os.path.join(current_dir, 'models')
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    classifier = BrainTumorClassifier(
        img_size=(224, 224),
        num_classes=num_classes,
        base_model_name='MobileNetV2',
        models_dir=models_dir
    )
    classifier.class_names = class_names
    
    # Load the original model as pre-trained model (not the retrained one)
    original_model_path = os.path.join(models_dir, 'brain_tumor_model.h5')
    
    # Determine path for saving retrained model
    if model_save_path.startswith('models/'):
        retrained_model_filename = os.path.basename(model_save_path)
        retrained_model_full_path = os.path.join(models_dir, retrained_model_filename)
    else:
        retrained_model_filename = model_save_path
        retrained_model_full_path = os.path.join(models_dir, retrained_model_filename) if not os.path.isabs(model_save_path) else model_save_path
    
    # Load original model as pre-trained model
    if os.path.exists(original_model_path):
        try:
            print(f"Loading original model from {original_model_path} as pre-trained model...")
            # Load original model to check number of classes
            temp_model = tf.keras.models.load_model(original_model_path)
            
            # Check if number of classes matches
            old_num_classes = temp_model.output_shape[-1]
            print(f"Existing model has {old_num_classes} output classes")
            print(f"New data has {num_classes} classes: {class_names}")
            
            if old_num_classes != num_classes:
                print(f"Warning: Class count mismatch! Existing model: {old_num_classes}, New data: {num_classes}")
                print("This can happen if the model was trained with 'unknown' class.")
                print("Attempting to handle class mismatch...")
                
                # If old model has 5 classes (with unknown) and new data has 4, rebuild with 4 classes
                # We can't use a 5-class model with 4-class data due to shape mismatch
                if old_num_classes == 5 and num_classes == 4 and 'unknown' not in class_names:
                    print("Model has 5 classes (includes 'unknown'), but data has 4 classes.")
                    print("Rebuilding model with 4 classes to match data...")
                    # Update classifier to use correct number of classes
                    classifier.num_classes = num_classes
                    classifier.build_model()
                    # Try to transfer weights from base model (excluding output layer)
                    try:
                        print("Transferring weights from old model's base layers...")
                        old_base = temp_model.layers[1]  # Base model layer
                        new_base = classifier.model.layers[1]  # New base model layer
                        # Copy weights from matching layers in base model
                        for old_layer, new_layer in zip(old_base.layers, new_base.layers):
                            if len(old_layer.get_weights()) > 0:
                                try:
                                    if len(new_layer.get_weights()) == len(old_layer.get_weights()):
                                        new_layer.set_weights(old_layer.get_weights())
                                except Exception as e:
                                    pass  # Skip layers that can't be transferred
                        print("Base model weights transferred successfully.")
                    except Exception as e:
                        print(f"Could not transfer weights: {str(e)}")
                        print("Starting with fresh model weights...")
                else:
                    print("Class mismatch cannot be handled automatically. Rebuilding model...")
                    classifier.num_classes = num_classes
                    classifier.build_model()
            else:
                # Classes match, use existing model
                classifier.model = temp_model
                # Unfreeze some layers for fine-tuning
                for layer in classifier.model.layers[1].layers[-4:]:
                    layer.trainable = True
                # Recompile to ensure metrics are built
                from tensorflow.keras.optimizers import Adam
                classifier.model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                print("Existing model loaded and prepared for fine-tuning.")
        except Exception as e:
            print(f"Could not load existing model: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Building new model...")
            classifier.build_model()
    else:
        print("No existing model found. Building new model...")
        classifier.build_model()
    
    # Step 5: Retrain the model (fine-tune on new data only)
    print(f"\nStep 5: Fine-tuning model on newly uploaded data (epochs={epochs}, fine_tune_epochs={fine_tune_epochs})...")
    print("Using existing model as pre-trained model for transfer learning.")
    try:
        # Ensure model is compiled before training
        if classifier.model is None:
            print("Error: Model is None. Cannot proceed with training.")
            raise ValueError("Model is None")
        
        # Verify model is compiled
        if not hasattr(classifier.model, 'optimizer') or classifier.model.optimizer is None:
            print("Model not compiled. Compiling now...")
            from tensorflow.keras.optimizers import Adam
            classifier.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        print(f"Starting training with {train_gen.samples} training samples and {val_gen.samples} validation samples...")
        # Use retrained model path for checkpoint to preserve original model
        classifier.train(
            train_gen,
            val_gen,
            epochs=epochs,
            fine_tune_epochs=fine_tune_epochs,
            checkpoint_path=retrained_model_filename  # Save checkpoints to retrained model path, not original
        )
        print("Model fine-tuning completed.")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up temp directory on error
        if os.path.exists(temp_data_dir):
            try:
                shutil.rmtree(temp_data_dir)
            except:
                pass
        db.update_training_session(training_session_id, status='failed', 
                                   notes=f"Training error: {str(e)}")
        return False
    
    # Step 6: Evaluate and save
    print("\nStep 6: Evaluating model...")
    final_metrics = None
    try:
        results = classifier.evaluate(val_gen)
        final_metrics = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        }
        
        print(f"\n=== Retraining Evaluation Results ===")
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Newly uploaded images used: {image_count}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        
        # Warn about suspiciously high accuracy on small datasets
        if results['accuracy'] >= 1.0 and val_gen.samples < 20:
            print(f"\n⚠️  WARNING: 100% accuracy achieved with only {val_gen.samples} validation samples!")
            print(f"   This is likely due to overfitting on a very small dataset.")
            print(f"   For reliable results, retrain with at least 20-30 images per class.")
            print(f"   Current dataset: {image_count} newly uploaded images across {num_classes} classes.")
            print(f"   Recommendation: Upload more images before deploying this model.")
        
        elif results['accuracy'] >= 0.95 and val_gen.samples < 10:
            print(f"\n⚠️  WARNING: Very high accuracy ({results['accuracy']:.2%}) with only {val_gen.samples} validation samples.")
            print(f"   This may indicate overfitting. Consider uploading more images for better generalization.")
        
        # Plot results
        classifier.plot_training_history('models/visualizations/training_history_retrain.png')
        classifier.plot_confusion_matrix(
            results['confusion_matrix'],
            results['class_names'],
            'models/visualizations/confusion_matrix_retrain.png'
        )
    except Exception as e:
        print(f"Warning: Evaluation failed: {str(e)}")
    
    # Save retrained model with new name
    print(f"\nStep 7: Saving retrained model to {retrained_model_full_path}...")
    print(f"Original model preserved at: {original_model_path}")
    try:
        classifier.save_model(retrained_model_filename)
        
        # Save class names
        class_names_path = os.path.join(models_dir, 'class_names.pkl')
        with open(class_names_path, 'wb') as f:
            pickle.dump(class_names, f)
        print(f"Class names saved to {class_names_path}")
        
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        db.update_training_session(training_session_id, status='failed', 
                                   notes=f"Error saving model: {str(e)}")
        return False
    
    # Step 8: Update database with final results
    print("\nStep 8: Updating database with training results...")
    db.update_training_session(
        session_id=training_session_id,
        status='completed',
        final_metrics=final_metrics,
        images_used=len(image_ids) if image_ids else images_processed
    )
    print("Database updated successfully.")
    
    # Step 9: Clean up temporary directory
    print("\nStep 9: Cleaning up temporary retraining directory...")
    try:
        if os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir)
            print(f"Temporary directory {temp_data_dir} removed successfully.")
    except Exception as e:
        print(f"Warning: Failed to remove temporary directory: {str(e)}")
    
    # Keep retrain_uploads directory files for reference and future retraining
    # Files are tracked in database, so they can be kept for audit trail
    print(f"\nNote: Uploaded files preserved in {retrain_data_dir} for reference.")
    print("Files are tracked in database and can be used for future retraining if needed.")
    
    print("\n" + "=" * 50)
    print("Model Retraining Process Completed Successfully!")
    print(f"Retrained on {image_count} newly uploaded images from database")
    print(f"Original model: models/brain_tumor_model.h5 (preserved)")
    print(f"Retrained model: {model_save_path} (saved)")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Default parameters
    epochs = 10
    fine_tune_epochs = 3
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    if len(sys.argv) > 2:
        fine_tune_epochs = int(sys.argv[2])
    
    # Run retraining
    success = retrain_model(
        retrain_data_dir='data/retrain_uploads',
        epochs=epochs,
        fine_tune_epochs=fine_tune_epochs
    )
    
    if success:
        print("\n✅ Retraining completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Retraining failed!")
        sys.exit(1)

