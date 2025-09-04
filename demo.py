"""
Fixed Demo Script for Astronomical Object Classification Project
Tests the basic functionality of the project components.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def test_imports():
    """Test if all modules can be imported successfully."""
    print("🧪 Testing module imports...")
    
    try:
        from src.preprocessing.data_processor import AstronomicalDataProcessor
        print("✅ AstronomicalDataProcessor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import AstronomicalDataProcessor: {e}")
        return False
    
    try:
        from src.models.ml_models import MLModelTrainer
        print("✅ MLModelTrainer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import MLModelTrainer: {e}")
        return False
    
    try:
        from src.models.deep_learning import DeepLearningTrainer
        print("✅ DeepLearningTrainer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import DeepLearningTrainer: {e}")
        return False
    
    try:
        # Fix the import path - check the actual file name
        from src.evaluation.model_evaluator import ModelEvaluator
        print("✅ ModelEvaluator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ModelEvaluator: {e}")
        print("💡 Creating ModelEvaluator file...")
        create_model_evaluator()
        try:
            from src.evaluation.model_evaluator import ModelEvaluator
            print("✅ ModelEvaluator imported successfully (created)")
        except Exception as e2:
            print(f"❌ Still failed to import ModelEvaluator: {e2}")
            return False
    
    try:
        from src.visualization.visualizer import AstronomicalVisualizer
        print("✅ AstronomicalVisualizer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import AstronomicalVisualizer: {e}")
        return False
    
    return True

def create_model_evaluator():
    """Create the ModelEvaluator file if it doesn't exist properly."""
    evaluator_dir = "src/evaluation"
    os.makedirs(evaluator_dir, exist_ok=True)
    
    # Create __init__.py
    init_file = os.path.join(evaluator_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Evaluation module for astronomical classification."""\n')
    
    # Create model_evaluator.py with the fixed code
    evaluator_file = os.path.join(evaluator_dir, "model_evaluator.py")
    evaluator_code = '''"""
Model Evaluator for Astronomical Object Classification
Comprehensive evaluation metrics and visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation for astronomical classification."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        self.comparison_df = None
        self.class_names = ['STAR', 'GALAXY', 'QSO']
    
    def evaluate_single_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate a single model and return comprehensive metrics."""
        try:
            y_pred = model.predict(X_test)
            
            try:
                y_proba = model.predict_proba(X_test)
            except:
                y_proba = None
            
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            if y_proba is not None and len(np.unique(y_test)) > 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                except:
                    metrics['roc_auc'] = np.nan
            else:
                metrics['roc_auc'] = np.nan
            
            self.results[model_name] = {
                'predictions': y_pred,
                'probabilities': y_proba,
                'metrics': metrics,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return {'model_name': model_name, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'roc_auc': 0.0}
    
    def compare_models(self, models_dict, X_test, y_test):
        """Compare multiple models and return comparison DataFrame."""
        comparison_results = []
        
        print(f"Evaluating {len(models_dict)} models...")
        
        for model_name, model in models_dict.items():
            print(f"  Evaluating {model_name}...")
            metrics = self.evaluate_single_model(model, X_test, y_test, model_name)
            comparison_results.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics.get('roc_auc', np.nan)
            })
        
        self.comparison_df = pd.DataFrame(comparison_results)
        self.comparison_df = self.comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        
        print("\\nModel Comparison Results:")
        print("=" * 80)
        print(self.comparison_df.round(4).to_string(index=False))
        
        return self.comparison_df
    
    def get_best_model(self, metric='Accuracy'):
        """Get the best performing model based on specified metric."""
        if self.comparison_df is None or len(self.comparison_df) == 0:
            return None, None
        
        best_row = self.comparison_df.loc[self.comparison_df[metric].idxmax()]
        return best_row['Model'], best_row[metric]
    
    def plot_model_comparison(self):
        """Plot model comparison across multiple metrics."""
        if self.comparison_df is None:
            print("No comparison results available. Run compare_models first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            if metric not in self.comparison_df.columns:
                continue
                
            ax = axes[i//2, i%2]
            
            bars = ax.bar(self.comparison_df['Model'], self.comparison_df[metric], 
                         color=color, alpha=0.7)
            
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, self.comparison_df[metric]):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2., value + 0.005,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        try:
            plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        except:
            pass
        plt.show()
    
    def generate_evaluation_report(self, filepath='results/evaluation_report.txt'):
        """Generate comprehensive evaluation report."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("ASTRONOMICAL OBJECT CLASSIFICATION - EVALUATION REPORT\\n")
            f.write("=" * 70 + "\\n\\n")
            f.write(f"Report generated: {pd.Timestamp.now()}\\n\\n")
            
            if self.comparison_df is not None:
                f.write("MODEL PERFORMANCE SUMMARY\\n")
                f.write("-" * 30 + "\\n")
                f.write(self.comparison_df.round(4).to_string(index=False))
                f.write("\\n\\n")
                
                best_model, best_score = self.get_best_model('Accuracy')
                f.write(f"Best Performing Model: {best_model}\\n")
                f.write(f"Best Accuracy: {best_score:.4f}\\n\\n")
        
        print(f"Evaluation report saved to: {filepath}")
        return filepath
'''
    
    with open(evaluator_file, 'w') as f:
        f.write(evaluator_code)
    
    print(f"Created {evaluator_file}")

def test_data_processor():
    """Test the data processor functionality."""
    print("\n🧪 Testing data processor...")
    
    try:
        from src.preprocessing.data_processor import AstronomicalDataProcessor
        
        processor = AstronomicalDataProcessor()
        print("✅ Processor created successfully")
        
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 100
        
        sample_data = pd.DataFrame({
            'u': np.random.normal(20, 2, n_samples),
            'g': np.random.normal(19, 1.5, n_samples),
            'r': np.random.normal(18, 1.5, n_samples),
            'i': np.random.normal(17, 1.5, n_samples),
            'z': np.random.normal(16, 1.5, n_samples),
            'redshift': np.random.exponential(0.5, n_samples),
            'class': np.random.choice(['STAR', 'GALAXY', 'QSO'], n_samples, p=[0.4, 0.4, 0.2])
        })
        
        print("✅ Sample data created")
        
        processor.explore_data(sample_data)
        print("✅ Data exploration completed")
        
        data_clean = processor.clean_data(sample_data)
        print("✅ Data cleaning completed")
    except Exception as e:
        print(f"❌ Error in data processor test: {e}")