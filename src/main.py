"""
Main Execution Script for Astronomical Object Classification
Orchestrates the entire ML pipeline from data loading to model evaluation.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_processor import AstronomicalDataProcessor
from src.models.ml_models import MLModelTrainer
from src.models.deep_learning import DeepLearningTrainer
from src.evaluation.model_evaluator import ModelEvaluator
from src.visualization.visualizer import EnhancedAstronomicalVisualizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def main():
    """
    Main execution function for the astronomical classification pipeline.
    """
    print("🌟" * 20)
    print("ASTRONOMICAL OBJECT CLASSIFICATION PIPELINE")
    print("🌟" * 20)
    print()
    
    # Initialize components
    processor = AstronomicalDataProcessor()
    ml_trainer = MLModelTrainer()
    dl_trainer = DeepLearningTrainer(num_classes=3)
    evaluator = ModelEvaluator()
    visualizer = EnhancedAstronomicalVisualizer()
    
    # Configuration
    config = {
        'data_path': 'data/Skyserver_SQL2_27_2018_6_51_39_PM.csv',
        'target_col': 'class',
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'deep_learning_epochs': 50,
        'deep_learning_batch_size': 32
    }
    
    print("📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Step 1: Data Loading and Exploration
    print("=" * 60)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(config['data_path']):
        print(f"❌ Data file not found: {config['data_path']}")
        print("Please place your Skyserver dataset in the data/ directory.")
        print("Expected filename: Skyserver_SQL2_27_2018_6_51_39_PM.csv")
        return
    
    # Load data
    data = processor.load_data(config['data_path'])
    if data is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Explore data
    processor.explore_data(data)
    
    # Create visualizations
    print("\n📊 Creating initial data visualizations...")
    visualizer.plot_data_distribution(data, config['target_col'])
    visualizer.plot_correlation_matrix(data)
    
    # Step 2: Data Preprocessing
    print("\n" + "=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)
    
    # Clean data
    data_clean = processor.clean_data(data)
    
    # Engineer features
    data_engineered = processor.engineer_features(data_clean)
    
    # Prepare features for ML
    X, y = processor.prepare_features(data_engineered, config['target_col'])
    
    # Scale features
    X_scaled = processor.scale_features(X, method='standard')
    
    # Feature selection
    X_selected = processor.select_features(X_scaled, y, method='mutual_info', k=50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=config['test_size'], 
        random_state=config['random_state'], stratify=y
    )
    
    print(f"\n✅ Data preprocessing completed!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Step 3: Traditional Machine Learning
    print("\n" + "=" * 60)
    print("STEP 3: TRADITIONAL MACHINE LEARNING")
    print("=" * 60)
    
    # Train all ML models
    ml_models = ml_trainer.train_all_models(
        X_train, y_train, cv=config['cv_folds']
    )
    
    # Save ML models
    ml_trainer.save_models('models/')
    
    # Step 4: Deep Learning
    print("\n" + "=" * 60)
    print("STEP 4: DEEP LEARNING MODELS")
    print("=" * 60)
    
    # Train deep learning models
    dl_models = dl_trainer.train_all_models(
        X_train, y_train, 
        epochs=config['deep_learning_epochs'],
        batch_size=config['deep_learning_batch_size']
    )
    
    # Save DL models
    dl_trainer.save_models('models/')
    
    # Step 5: Model Evaluation
    print("\n" + "=" * 60)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 60)
    
    # Combine all models for evaluation
    all_models = {}
    all_models.update(ml_models)
    all_models.update(dl_models)
    
    # Evaluate all models
    comparison_df = evaluator.compare_models(all_models, X_test, y_test)
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report('results/evaluation_report.txt')
    
    # Step 6: Visualization and Analysis
    print("\n" + "=" * 60)
    print("STEP 6: VISUALIZATION AND ANALYSIS")
    print("=" * 60)
    
    # Create comprehensive visualizations
    print("📊 Creating model comparison visualizations...")
    evaluator.plot_model_comparison()
    
    print("📈 Creating confusion matrices...")
    evaluator.plot_confusion_matrices(evaluator.results)
    
    print("📊 Creating ROC curves...")
    evaluator.plot_roc_curves(evaluator.results)
    
    # Feature importance analysis
    print("🎯 Analyzing feature importance...")
    for model_name in ['random_forest', 'xgboost', 'lightgbm']:
        if model_name in ml_trainer.feature_importance:
            importance_df = ml_trainer.feature_importance[model_name]
            visualizer.plot_feature_importance(importance_df, top_n=20)
    
    # Dimensionality reduction visualization
    print("🔍 Creating dimensionality reduction visualizations...")
    visualizer.plot_dimensionality_reduction(X_selected, y, method='pca')
    
    # Astronomical features analysis
    print("🌟 Creating astronomical feature analysis...")
    visualizer.plot_astronomical_features(data_engineered, target_col=config['target_col'])
    
    # Step 7: Results Summary
    print("\n" + "=" * 60)
    print("STEP 7: RESULTS SUMMARY")
    print("=" * 60)
    
    # Get best model
    best_model_name, best_score = evaluator.get_best_model('accuracy')
    print(f"🏆 Best Model: {best_model_name}")
    print(f"🏆 Best Accuracy: {best_score:.4f}")
    
    # Model ranking
    print("\n📊 MODEL RANKING (by Accuracy):")
    print(comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].round(4))
    
    # Save results
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print(f"\n💾 Results saved to results/ directory")
    
    # Step 8: Interactive Dashboard
    print("\n" + "=" * 60)
    print("STEP 8: INTERACTIVE DASHBOARD")
    print("=" * 60)
    
    # Create interactive dashboard
    dashboard = visualizer.create_interactive_dashboard(data_engineered, config['target_col'])
    
    # Save dashboard as HTML
    if dashboard:
        dashboard.write_html('results/interactive_dashboard.html')
        print("💾 Interactive dashboard saved to results/interactive_dashboard.html")
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📁 Check the following directories for results:")
    print("   - models/: Trained models")
    print("   - results/: Evaluation reports and visualizations")
    print("   - data/: Processed datasets")
    print("\n🚀 Your astronomical classification project is ready!")

def run_quick_demo():
    """
    Run a quick demo with sample data if the main dataset is not available.
    """
    print("🚀 Running Quick Demo...")
    
    # Create sample astronomical data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic astronomical features
    data = pd.DataFrame({
        'u': np.random.normal(20, 2, n_samples),
        'g': np.random.normal(19, 1.5, n_samples),
        'r': np.random.normal(18, 1.5, n_samples),
        'i': np.random.normal(17, 1.5, n_samples),
        'z': np.random.normal(16, 1.5, n_samples),
        'redshift': np.random.exponential(0.5, n_samples),
        'class': np.random.choice(['STAR', 'GALAXY', 'QSO'], n_samples, p=[0.4, 0.4, 0.2])
    })
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/sample_astronomical_data.csv', index=False)
    
    print("✅ Sample data created. Run the main pipeline with this data.")
    return data

if __name__ == "__main__":
    try:
        # Check if data exists, otherwise run demo
        if not os.path.exists('data/Skyserver_SQL2_27_2018_6_51_39_PM.csv'):
            print("⚠️ Main dataset not found. Creating sample data for demo...")
            run_quick_demo()
            print("\n📝 To use your own data:")
            print("   1. Place your Skyserver dataset in the data/ directory")
            print("   2. Rename it to: Skyserver_SQL2_27_2018_6_51_39_PM.csv")
            print("   3. Run this script again")
            print("\n🚀 Running demo with sample data...")
        
        main()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
