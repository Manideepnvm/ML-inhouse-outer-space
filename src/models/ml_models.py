import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
from tqdm import tqdm

# XGBoost and LightGBM imports with error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available")

warnings.filterwarnings('ignore')

class MLModelTrainer:
    """
    Comprehensive ML model trainer for astronomical classification.
    """
    
    def __init__(self):
        """Initialize the ML model trainer."""
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
    def get_model_configs(self):
        """
        Get model configurations with hyperparameters.
        
        Returns:
            dict: Model configurations
        """
        configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            },
            'knn': {
                'model': KNeighborsClassifier(n_jobs=-1),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [5, 10, 20, 30, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'criterion': ['gini', 'entropy']
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'model': xgb.XGBClassifier(
                    random_state=42, 
                    eval_metric='mlogloss',
                    verbosity=0
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            configs['lightgbm'] = {
                'model': lgb.LGBMClassifier(
                    random_state=42,
                    verbosity=-1,
                    force_col_wise=True
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [5, 10, 15],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            configs['catboost'] = {
                'model': cb.CatBoostClassifier(
                    random_state=42,
                    verbose=False
                ),
                'params': {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3, 5]
                }
            }
        
        return configs
    
    def train_single_model(self, model_name, X_train, y_train, cv=5, tune_hyperparams=True):
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            cv (int): Cross-validation folds
            tune_hyperparams (bool): Whether to tune hyperparameters
            
        Returns:
            object: Trained model
        """
        print(f"\n🚀 Training {model_name.upper()}")
        print("-" * 40)
        
        configs = self.get_model_configs()
        
        if model_name not in configs:
            print(f"❌ Model '{model_name}' not available")
            return None
        
        config = configs[model_name]
        model = config['model']
        
        if tune_hyperparams and len(config['params']) > 0:
            # Use GridSearchCV for hyperparameter tuning
            search = GridSearchCV(model, config['params'], cv=cv, scoring='accuracy', n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            self.best_params[model_name] = search.best_params_
            print(f"   ✅ Best params: {search.best_params_}")
            print(f"   ✅ Best CV score: {search.best_score_:.4f}")
        else:
            print(f"   🏃 Training with default parameters...")
            model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        self.cv_scores[model_name] = cv_scores
        
        print(f"   📊 CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.models[model_name] = model
        return model
    
    def train_all_models(self, X_train, y_train, cv=5, tune_hyperparams=True):
        """
        Train all available models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            cv (int): Cross-validation folds
            tune_hyperparams (bool): Whether to tune hyperparameters
            
        Returns:
            dict: Dictionary of trained models
        """
        print("\n🤖 TRAINING ALL MACHINE LEARNING MODELS")
        print("=" * 60)
        
        configs = self.get_model_configs()
        
        for model_name in tqdm(configs.keys(), desc="Training models"):
            try:
                self.train_single_model(model_name, X_train, y_train, cv, tune_hyperparams)
                print(f"✅ {model_name} trained successfully!")
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue
        
        print(f"\n🎉 Training completed! {len(self.models)} models trained.")
        return self.models
    
    def predict(self, model_name, X_test):
        """
        Make predictions using a trained model.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train the model first.")
        
        return self.models[model_name].predict(X_test)
    
    def predict_proba(self, model_name, X_test):
        """
        Get prediction probabilities.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train the model first.")
        
        model = self.models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_test)
        else:
            print(f"⚠️ Model '{model_name}' doesn't support probability predictions")
            return None
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a single model.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'precision_weighted': report['weighted avg']['precision'],
            'recall_weighted': report['weighted avg']['recall'],
            'f1_weighted': report['weighted avg']['f1-score']
        }
        
        return metrics
    
    def get_feature_importance_summary(self):
        """
        Get feature importance summary across all tree-based models.
        
        Returns:
            pd.DataFrame: Combined feature importance
        """
        if not self.feature_importance:
            print("⚠️ No feature importance data available")
            return pd.DataFrame()
        
        # Combine feature importance from all models
        combined_importance = pd.DataFrame()
        
        for model_name, importance_df in self.feature_importance.items():
            if combined_importance.empty:
                combined_importance = importance_df.copy()
                combined_importance.rename(columns={'importance': f'{model_name}_importance'}, inplace=True)
            else:
                combined_importance = combined_importance.merge(
                    importance_df[['feature', 'importance']].rename(
                        columns={'importance': f'{model_name}_importance'}
                    ),
                    on='feature', how='outer'
                )
        
        # Calculate mean importance
        importance_cols = [col for col in combined_importance.columns if 'importance' in col]
        combined_importance['mean_importance'] = combined_importance[importance_cols].mean(axis=1)
        combined_importance = combined_importance.sort_values('mean_importance', ascending=False)
        
        return combined_importance
    
    def save_models(self, save_dir='models/'):
        """
        Save all trained models to disk.
        
        Args:
            save_dir (str): Directory to save models
        """
        print(f"\n💾 SAVING MODELS TO {save_dir}")
        print("-" * 40)
        
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                model_path = os.path.join(save_dir, f'{model_name}_model.joblib')
                joblib.dump(model, model_path)
                print(f"   ✅ {model_name} saved to {model_path}")
            except Exception as e:
                print(f"   ❌ Error saving {model_name}: {e}")
        
        # Save additional metadata
        metadata = {
            'best_params': self.best_params,
            'cv_scores': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in self.cv_scores.items()}
        }
        
        metadata_path = os.path.join(save_dir, 'ml_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        print(f"   ✅ Metadata saved to {metadata_path}")
        
        # Save feature importance
        if self.feature_importance:
            importance_path = os.path.join(save_dir, 'feature_importance.joblib')
            joblib.dump(self.feature_importance, importance_path)
            print(f"   ✅ Feature importance saved to {importance_path}")
    
    def load_models(self, load_dir='models/'):
        """
        Load saved models from disk.
        
        Args:
            load_dir (str): Directory to load models from
        """
        print(f"\n📂 LOADING MODELS FROM {load_dir}")
        print("-" * 40)
        
        if not os.path.exists(load_dir):
            print(f"❌ Model directory not found: {load_dir}")
            return
        
        # Load models
        model_files = [f for f in os.listdir(load_dir) if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.joblib', '')
            model_path = os.path.join(load_dir, model_file)
            
            try:
                self.models[model_name] = joblib.load(model_path)
                print(f"   ✅ {model_name} loaded")
            except Exception as e:
                print(f"   ❌ Error loading {model_name}: {e}")
        
        # Load metadata
        metadata_path = os.path.join(load_dir, 'ml_metadata.joblib')
        if os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                self.best_params = metadata.get('best_params', {})
                self.cv_scores = metadata.get('cv_scores', {})
                print(f"   ✅ Metadata loaded")
            except Exception as e:
                print(f"   ⚠️ Error loading metadata: {e}")
        
        # Load feature importance
        importance_path = os.path.join(load_dir, 'feature_importance.joblib')
        if os.path.exists(importance_path):
            try:
                self.feature_importance = joblib.load(importance_path)
                print(f"   ✅ Feature importance loaded")
            except Exception as e:
                print(f"   ⚠️ Error loading feature importance: {e}")
    
    def get_model_summary(self):
        """
        Get a summary of all trained models.
        
        Returns:
            pd.DataFrame: Model summary
        """
        if not self.models:
            print("⚠️ No models trained yet")
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, model in self.models.items():
            cv_score = self.cv_scores.get(model_name, [0])
            cv_mean = np.mean(cv_score) if isinstance(cv_score, (list, np.ndarray)) else cv_score
            cv_std = np.std(cv_score) if isinstance(cv_score, (list, np.ndarray)) else 0
            
            summary_data.append({
                'Model': model_name,
                'Type': type(model).__name__,
                'CV_Score_Mean': cv_mean,
                'CV_Score_Std': cv_std,
                'Parameters_Tuned': model_name in self.best_params,
                'Feature_Importance': hasattr(model, 'feature_importances_')
            })
        
        return pd.DataFrame(summary_data).sort_values('CV_Score_Mean', ascending=False)
    
    def quick_train(self, X_train, y_train, model_list=None):
        """
        Quick training without hyperparameter tuning for fast prototyping.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            model_list (list): List of models to train (None = all)
            
        Returns:
            dict: Trained models
        """
        print("\n⚡ QUICK TRAINING MODE")
        print("-" * 40)
        
        configs = self.get_model_configs()
        
        if model_list is None:
            model_list = list(configs.keys())
        
        for model_name in model_list:
            if model_name in configs:
                try:
                    model = configs[model_name]['model']
                    model.fit(X_train, y_train)
                    self.models[model_name] = model
                    print(f"   ✅ {model_name} trained")
                except Exception as e:
                    print(f"   ❌ Error training {model_name}: {e}")
        
        return self.models
    
    def ensemble_predict(self, X_test, method='voting'):
        """
        Make ensemble predictions using all trained models.
        
        Args:
            X_test (np.ndarray): Test features
            method (str): Ensemble method ('voting', 'weighted')
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models trained yet")
        
        print(f"\n🎭 ENSEMBLE PREDICTION ({method})")
        print("-" * 40)
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            try:
                predictions[model_name] = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    probabilities[model_name] = model.predict_proba(X_test)
            except Exception as e:
                print(f"   ⚠️ Error getting predictions from {model_name}: {e}")
        
        if method == 'voting':
            # Simple majority voting
            pred_df = pd.DataFrame(predictions)
            ensemble_pred = pred_df.mode(axis=1)[0].values
            
        elif method == 'weighted' and probabilities:
            # Weighted average of probabilities
            weights = [self.cv_scores.get(name, [0.5])[0] if isinstance(self.cv_scores.get(name, [0.5]), list) 
                      else self.cv_scores.get(name, 0.5) for name in probabilities.keys()]
            
            if sum(weights) > 0:
                weights = np.array(weights) / sum(weights)  # Normalize weights
                
                # Average probabilities
                avg_proba = np.zeros_like(list(probabilities.values())[0])
                for i, (model_name, proba) in enumerate(probabilities.items()):
                    avg_proba += weights[i] * proba
                
                ensemble_pred = np.argmax(avg_proba, axis=1)
            else:
                # Fallback to simple voting
                pred_df = pd.DataFrame(predictions)
                ensemble_pred = pred_df.mode(axis=1)[0].values
        else:
            # Fallback to simple voting
            pred_df = pd.DataFrame(predictions)
            ensemble_pred = pred_df.mode(axis=1)[0].values
        
        print(f"   ✅ Ensemble predictions generated using {len(predictions)} models")
        return ensemble_pred
    
    def get_model_complexity(self):
        """
        Analyze model complexity and training time.
        
        Returns:
            pd.DataFrame: Model complexity analysis
        """
        complexity_data = []
        
        for model_name, model in self.models.items():
            complexity = {
                'Model': model_name,
                'Parameters': self._count_parameters(model),
                'Memory_Usage_MB': self._estimate_memory_usage(model),
                'Interpretable': self._is_interpretable(model)
            }
            complexity_data.append(complexity)
        
        return pd.DataFrame(complexity_data)
    
    def _count_parameters(self, model):
        """Count model parameters (approximation)."""
        if hasattr(model, 'coef_'):
            return np.prod(model.coef_.shape)
        elif hasattr(model, 'n_features_in_'):
            return getattr(model, 'n_estimators', 1) * model.n_features_in_
        else:
            return 'Unknown'
    
    def _estimate_memory_usage(self, model):
        """Estimate model memory usage in MB."""
        try:
            import pickle
            return len(pickle.dumps(model)) / (1024 * 1024)
        except:
            return 'Unknown'
    
    def _is_interpretable(self, model):
        """Check if model is interpretable."""
        interpretable_models = [
            'LogisticRegression', 
            'DecisionTreeClassifier',
            'GaussianNB'
        ]
        return type(model).__name__ in interpretable_models
    
    def create_model_comparison_chart(self):
        """
        Create a comparison chart of model performance.
        
        Returns:
            pd.DataFrame: Comparison data for visualization
        """
        if not self.cv_scores:
            print("⚠️ No CV scores available for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, scores in self.cv_scores.items():
            if isinstance(scores, (list, np.ndarray)):
                mean_score = np.mean(scores)
                std_score = np.std(scores)
            else:
                mean_score = scores
                std_score = 0
            
            comparison_data.append({
                'Model': model_name,
                'Mean_CV_Score': mean_score,
                'Std_CV_Score': std_score,
                'Min_Score': mean_score - std_score,
                'Max_Score': mean_score + std_score
            })
        
        return pd.DataFrame(comparison_data).sort_values('Mean_CV_Score', ascending=False)
    
    def hyperparameter_importance_analysis(self):
        """
        Analyze which hyperparameters had the biggest impact.
        
        Returns:
            dict: Hyperparameter importance analysis
        """
        if not self.best_params:
            print("⚠️ No hyperparameter tuning results available")
            return {}
        
        print("\n🎯 HYPERPARAMETER ANALYSIS")
        print("-" * 40)
        
        for model_name, params in self.best_params.items():
            print(f"\n{model_name.upper()}:")
            for param, value in params.items():
                print(f"   {param}: {value}")
        
        return self.best_params
            
