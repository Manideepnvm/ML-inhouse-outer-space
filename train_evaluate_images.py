
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.models.deep_learning import ImageClassifier

def load_dataset(data_dir):
    """Load images from Galaxy, quasers, stars folders"""
    images = []
    labels = []
    class_names = ['STAR', 'GALAXY', 'QSO']
    
    # Map folder names to class names
    # Folder structure: data/Galaxy, data/quasers, data/stars
    # Target classes: STAR, GALAXY, QSO
    
    mapping = {
        'stars': 'STAR', 
        'Galaxy': 'GALAXY', 
        'quasers': 'QSO'
    }
    
    img_size = (64, 64)
    
    print("Loading images...")
    
    for folder, class_name in mapping.items():
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found")
            continue
            
        label_idx = class_names.index(class_name)
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        print(f"  Found {len(files)} images in {folder} ({class_name})")
        
        for file in files:
            img_path = os.path.join(folder_path, file)
            try:
                img = load_img(img_path, target_size=img_size)
                img_arr = img_to_array(img)
                images.append(img_arr)
                labels.append(label_idx)
            except Exception as e:
                print(f"  Error loading {file}: {e}")
                
    X = np.array(images)
    y = np.array(labels)
    
    # Normalize
    X = X / 255.0
    
    # One-hot encode labels
    y_encoded = tf.keras.utils.to_categorical(y, num_classes=3)
    
    return X, y_encoded, class_names

def run_image_analysis(data_dir="data"):
    """
    Run the complete image analysis pipeline: load, train, evaluate.
    Returns:
        str: Evaluation report content
    """
    print("\n" + "=" * 60)
    print("STEP: IMAGE ANALYSIS")
    print("=" * 60)
    
    # Ensure absolute path for data_dir
    if not os.path.isabs(data_dir):
        # Assuming data_dir is relative to project root
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, data_dir)
        
    X, y, class_names = load_dataset(data_dir)
    
    if len(X) == 0:
        msg = "No images found for image analysis!"
        print(msg)
        return msg
        
    print(f"Total images loaded: {len(X)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    
    # Initialize and Train
    classifier = ImageClassifier()
    model = classifier.build_model()
    
    if model is None:
        msg = "Failed to build image model"
        print(msg)
        return msg
        
    print("Training image model...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=8,
        validation_split=0.1,
        verbose=1
    )
    
    # Save trained model
    models_dir = os.path.join(os.path.dirname(data_dir), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, 'image_model.h5'))
    print("Image model saved to models/image_model.h5")
    
    # Evaluate
    print("Evaluating image model...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred)
    
    # Save Metrics to JSON for Dashboard
    import json
    report_dict = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    metrics_data = {
        'accuracy': history.history['accuracy'][-1],
        'loss': history.history['loss'][-1],
        'test_accuracy': report_dict['accuracy'],
        'report': report_dict
    }
    
    with open('results/image_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("Metrics saved to results/image_metrics.json")
    
    # Generate Report Content
    report_content = f"""
============================================================
IMAGE ANALYSIS EVALUATION
============================================================
Date: {pd.Timestamp.now()}
Total Images: {len(X)}
Classes: {', '.join(class_names)}

## Model Performance (Test Set)
{report}

## Confusion Matrix
{conf_mat}

## Training History
Final Accuracy: {history.history['accuracy'][-1]:.4f}
Final Loss: {history.history['loss'][-1]:.4f}
"""
    return report_content

if __name__ == "__main__":
    import pandas as pd
    report = run_image_analysis()
    
    # Save standalone report if run directly
    with open('results/image_evaluation_report.txt', 'w') as f:
        f.write(report)
    print(f"\nReport generated: results/image_evaluation_report.txt")
    print(report)
