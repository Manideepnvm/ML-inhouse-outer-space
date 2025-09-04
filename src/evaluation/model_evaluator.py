"""
Model Evaluator for Astronomical Object Classification
Comprehensive evaluation metrics and visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score
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
        """
        Evaluate a single model and return comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            try:
                y_proba = model.predict_proba(X_test)
            except:
                y_proba = None
            
            # Calculate basic metrics
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Calculate per-class metrics
            per_class_metrics = {}
            for i, class_name in enumerate(self.class_names):
                if i < len(np.unique(y_test)):
                    per_class_metrics[f'{class_name.lower()}_precision'] = precision_score(
                        y_test, y_pred, labels=[i], average='micro', zero_division=0
                    )
                    per_class_metrics[f'{class_name.lower()}_recall'] = recall_score(
                        y_test, y_pred, labels=[i], average='micro', zero_division=0
                    )
                    per_class_metrics[f'{class_name.lower()}_f1'] = f1_score(
                        y_test, y_pred, labels=[i], average='micro', zero_division=0
                    )
            
            metrics.update(per_class_metrics)
            
            # Calculate AUC if probabilities available and multiclass
            if y_proba is not None and len(np.unique(y_test)) > 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                except:
                    metrics['roc_auc'] = np.nan
            elif y_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                except:
                    metrics['roc_auc'] = np.nan
            else:
                metrics['roc_auc'] = np.nan
            
            # Store detailed results
            self.results[model_name] = {
                'predictions': y_pred,
                'probabilities': y_proba,
                'metrics': metrics,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, 
                                                             target_names=self.class_names[:len(np.unique(y_test))],
                                                             zero_division=0)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return {'model_name': model_name, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'roc_auc': 0.0}
    
    def compare_models(self, models_dict, X_test, y_test):
        """
        Compare multiple models and return comparison DataFrame.
        
        Args:
            models_dict: Dictionary of model_name: model pairs
            X_test: Test features
            y_test: Test labels
            
        Returns:
            pd.DataFrame: Comparison results sorted by accuracy
        """
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
        
        # Create comparison DataFrame
        self.comparison_df = pd.DataFrame(comparison_results)
        self.comparison_df = self.comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        
        # Display results
        print("\nModel Comparison Results:")
        print("=" * 80)
        print(self.comparison_df.round(4).to_string(index=False))
        
        return self.comparison_df
    
    def get_best_model(self, metric='Accuracy'):
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            tuple: (best_model_name, best_score)
        """
        if self.comparison_df is None or len(self.comparison_df) == 0:
            return None, None
        
        best_row = self.comparison_df.loc[self.comparison_df[metric].idxmax()]
        return best_row['Model'], best_row[metric]
    
    def plot_confusion_matrices(self, models_to_plot=None, figsize=(15, 10)):
        """
        Plot confusion matrices for specified models.
        
        Args:
            models_to_plot: List of model names to plot (None for all)
            figsize: Figure size
        """
        if not self.results:
            print("No evaluation results available. Run compare_models first.")
            return
        
        if models_to_plot is None:
            models_to_plot = list(self.results.keys())
        
        n_models = len(models_to_plot)
        if n_models == 0:
            return
        
        # Calculate subplot arrangement
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, model_name in enumerate(models_to_plot):
            if model_name not in self.results:
                continue
                
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            cm = self.results[model_name]['confusion_matrix']
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.class_names[:cm.shape[1]],
                       yticklabels=self.class_names[:cm.shape[0]])
            ax.set_title(f'{model_name}\nAccuracy: {self.results[model_name]["metrics"]["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, metrics=['Accuracy', 'Precision', 'Recall', 'F1-Score'], figsize=(15, 10)):
        """
        Plot model comparison across multiple metrics.
        
        Args:
            metrics: List of metrics to compare
            figsize: Figure size
        """
        if self.comparison_df is None:
            print("No comparison results available. Run compare_models first.")
            return
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']
        
        for i, (metric, color) in enumerate(zip(metrics[:4], colors)):
            if metric not in self.comparison_df.columns:
                continue
                
            ax = axes[i//2, i%2]
            
            # Create bar plot
            bars = ax.bar(self.comparison_df['Model'], self.comparison_df[metric], 
                         color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, self.comparison_df[metric]):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2., value + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, models_to_plot=None, figsize=(12, 8)):
        """
        Plot ROC curves for binary or multiclass classification.
        
        Args:
            models_to_plot: List of model names to plot
            figsize: Figure size
        """
        if not self.results:
            print("No evaluation results available.")
            return
        
        if models_to_plot is None:
            models_to_plot = list(self.results.keys())
        
        plt.figure(figsize=figsize)
        
        for model_name in models_to_plot:
            if model_name not in self.results:
                continue
                
            result = self.results[model_name]
            if result['probabilities'] is None:
                continue
            
            try:
                # For binary classification
                if result['probabilities'].shape[1] == 2:
                    y_test = [1 if pred == 1 else 0 for pred in result['predictions']]  # Simplified for demo
                    fpr, tpr, _ = roc_curve(y_test, result['probabilities'][:, 1])
                    auc = roc_auc_score(y_test, result['probabilities'][:, 1])
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
                
            except Exception as e:
                print(f"Could not plot ROC curve for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, filepath='results/evaluation_report.txt'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            filepath: Path to save the report
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("ASTRONOMICAL OBJECT CLASSIFICATION - EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Report generated: {pd.Timestamp.now()}\n\n")
            
            # Model comparison summary
            if self.comparison_df is not None:
                f.write("MODEL PERFORMANCE SUMMARY\n")
                f.write("-" * 30 + "\n")
                f.write(self.comparison_df.round(4).to_string(index=False))
                f.write("\n\n")
                
                # Best model
                best_model, best_score = self.get_best_model('Accuracy')
                f.write(f"Best Performing Model: {best_model}\n")
                f.write(f"Best Accuracy: {best_score:.4f}\n\n")
            
            # Detailed results for each model
            if self.results:
                f.write("DETAILED MODEL RESULTS\n")
                f.write("-" * 25 + "\n\n")
                
                for model_name, result in self.results.items():
                    f.write(f"Model: {model_name}\n")
                    f.write("=" * 20 + "\n")
                    
                    # Metrics
                    metrics = result['metrics']
                    f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                    f.write(f"Precision: {metrics['precision']:.4f}\n")
                    f.write(f"Recall: {metrics['recall']:.4f}\n")
                    f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
                    if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
                        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
                    f.write("\n")
                    
                    # Classification report
                    f.write("Classification Report:\n")
                    f.write(result['classification_report'])
                    f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"Evaluation report saved to: {filepath}")
        return filepath
    
    def cross_validate_models(self, models_dict, X, y, cv=5, scoring='accuracy'):
        """
        Perform cross-validation on multiple models.
        
        Args:
            models_dict: Dictionary of models
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            pd.DataFrame: Cross-validation results
        """
        cv_results = []
        
        print(f"Performing {cv}-fold cross-validation...")
        
        for model_name, model in models_dict.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                cv_results.append({
                    'Model': model_name,
                    'CV_Mean': scores.mean(),
                    'CV_Std': scores.std(),
                    'CV_Min': scores.min(),
                    'CV_Max': scores.max()
                })
                print(f"  {model_name}: {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                print(f"  CV failed for {model_name}: {e}")
                cv_results.append({
                    'Model': model_name,
                    'CV_Mean': 0.0,
                    'CV_Std': 0.0,
                    'CV_Min': 0.0,
                    'CV_Max': 0.0
                })
        
        cv_df = pd.DataFrame(cv_results).sort_values('CV_Mean', ascending=False)
        return cv_df
    
    def feature_importance_analysis(self, model, feature_names, model_name="Model"):
        """
        Analyze feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} does not have feature importance.")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_results(self, directory='results'):
        """
        Save all evaluation results to files.
        
        Args:
            directory: Directory to save results
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save comparison results
        if self.comparison_df is not None:
            self.comparison_df.to_csv(f'{directory}/model_comparison.csv', index=False)
        
        # Save detailed results
        if self.results:
            import pickle
            with open(f'{directory}/detailed_results.pkl', 'wb') as f:
                pickle.dump(self.results, f)
        
        print(f"Results saved to {directory}/")