import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
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
        self.ensemble_models = {}
        self.scaler = StandardScaler()
        
    def get_model_configs(self):
        """
        Get enhanced model configurations with comprehensive hyperparameters.
        
        Returns:
            dict: Model configurations with advanced parameter grids
        """
        configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=2000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga', 'lbfgs'],
                    'class_weight': [None, 'balanced'],
                    'l1_ratio': [0.15, 0.5, 0.7, 0.9]  # For elasticnet
                },
                'description': """
                **Logistic Regression** - A probabilistic algorithm that models the probability of class 
                membership for Stars, Galaxies, and Quasars by applying the logistic function to linear 
                combinations of features. This algorithm:
                - Provides interpretable predictions for celestial object classification
                - Enables fast probabilistic identification of astronomical objects
                - Offers feature coefficient analysis for understanding object characteristics
                - Serves as a baseline for comparing advanced algorithm performance
                - Aligns with academic standards for linear classification methodologies
                """
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True),
                'params': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [5, 10, 15, 20, 25, 30, None],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                    'bootstrap': [True, False],
                    'class_weight': [None, 'balanced', 'balanced_subsample'],
                    'criterion': ['gini', 'entropy']
                },
                'description': """
                **Random Forest** - An ensemble learning algorithm that combines multiple decision trees 
                using bagging to achieve robust predictions for Stars, Galaxies, and Quasars classification. 
                This algorithm's working principles include:
                - Bootstrap aggregating (bagging) multiple decision trees for ensemble robustness
                - Feature importance ranking for identifying key astronomical characteristics
                - Non-linear decision boundary learning for complex celestial object patterns
                - Inherent handling of missing values and outliers in astronomical data
                - Out-of-bag (OOB) error estimation without requiring separate validation sets
                - Reduced overfitting through ensemble diversity, aligning with academic standards
                """
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2],
                    'max_depth': [2, 3, 4, 5, 6, 7, 8],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 6],
                    'criterion': ['friedman_mse', 'squared_error']
                },
                'description': """
                **Gradient Boosting** - A sequential ensemble algorithm that builds models iteratively, 
                with each new model correcting prediction errors of previous models. For celestial object 
                prediction, this algorithm:
                - Sequentially trains models to minimize residual errors in classifications
                - Achieves high accuracy through iterative error correction and refinement
                - Enables detailed feature importance analysis for astronomical data characteristics
                - Learns complex non-linear relationships between photometric features and object types
                - Implements learning rate control to balance model complexity and generalization
                - Adheres to academic standards for gradient-based optimization methodologies
                """
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
                },
                'description': """
                **Support Vector Machine (SVM)** - A powerful algorithm that finds optimal hyperplanes 
                to separate different classes with maximum margin. For celestial object prediction, SVM:
                - Identifies optimal decision boundaries separating Stars, Galaxies, and Quasars
                - Employs kernel methods (RBF, polynomial, sigmoid) to handle non-linear feature spaces
                - Maximizes margin between class boundaries for improved generalization
                - Effectively handles high-dimensional astronomical feature spaces
                - Provides robust probabilistic predictions for celestial object classification
                - Represents a well-established algorithm in academic machine learning literature
                """
            },
            'knn': {
                'model': KNeighborsClassifier(n_jobs=-1),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                'description': """
                **K-Nearest Neighbors (KNN)** - A non-parametric algorithm that classifies celestial objects 
                based on proximity of neighboring instances in the feature space. This algorithm:
                - Identifies the k nearest neighbors in feature space for each object
                - Assigns class labels based on majority voting among neighbors
                - Captures local patterns and proximities in astronomical data
                - Requires no explicit model training, enabling fast online learning
                - Employs various distance metrics (Euclidean, Manhattan, Minkowski) for flexibility
                - Aligns with academic standards for instance-based learning methodologies
                """
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
                },
                'description': """
                **Decision Tree** - A recursive partitioning algorithm that creates interpretable 
                decision rules for classifying Stars, Galaxies, and Quasars. This algorithm:
                - Recursively splits feature space based on optimal information gain/Gini impurity
                - Generates interpretable decision rules and tree structures for celestial object classification
                - Handles both categorical and continuous astronomical features natively
                - Provides clear visualization of decision-making processes for academic analysis
                - Captures non-linear relationships through hierarchical feature interactions
                - Enables feature importance ranking based on split decisions
                - Serves as the foundational algorithm for ensemble methods like Random Forest
                - Represents a fundamental algorithm in academic machine learning methodologies
                """
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
            # Use RandomizedSearchCV for faster hyperparameter tuning
            from sklearn.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(
                model, config['params'], 
                n_iter=5,  # Reduced from full grid search to 5 random combinations
                cv=cv, 
                scoring='accuracy', 
                n_jobs=-1,
                random_state=42
            )
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
    
    def train_quick_models(self, X_train, y_train, model_names=None, cv=3):
        """
        Train only the most effective models for faster execution.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            model_names (list): List of model names to train
            cv (int): Cross-validation folds
            
        Returns:
            dict: Dictionary of trained models
        """
        if model_names is None:
            model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
        
        print(f"\n🚀 QUICK MODE: Training {len(model_names)} best models")
        print("=" * 50)
        
        for model_name in model_names:
            try:
                # Train with minimal hyperparameter tuning for speed
                model = self.train_single_model(
                    model_name, X_train, y_train, cv=cv, tune_hyperparams=False
                )
                if model is not None:
                    self.models[model_name] = model
                    print(f"   ✅ {model_name} trained successfully")
            except Exception as e:
                print(f"   ❌ Error training {model_name}: {e}")
        
        print(f"\n🎉 Quick training completed! {len(self.models)} models trained.")
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
    
    def create_advanced_ensembles(self, X_train, y_train, X_val=None, y_val=None):
        """
        Create advanced ensemble models including stacking and voting.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            
        Returns:
            dict: Dictionary of ensemble models
        """
        print("\n🎭 CREATING ADVANCED ENSEMBLE MODELS")
        print("=" * 60)
        
        # Ensure we have some base models trained
        if len(self.models) < 2:
            print("⚠️ Need at least 2 base models for ensemble creation")
            return {}
        
        # Scale features for ensemble methods
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Create base estimators list
        base_estimators = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):  # Only models with probability predictions
                base_estimators.append((name, model))
        
        if len(base_estimators) < 2:
            print("⚠️ Need at least 2 models with probability predictions for ensemble")
            return {}
        
        # 1. Voting Classifier (Hard Voting)
        try:
            voting_hard = VotingClassifier(
                estimators=base_estimators,
                voting='hard'
            )
            voting_hard.fit(X_train_scaled, y_train)
            self.ensemble_models['voting_hard'] = voting_hard
            print("   ✅ Hard Voting Classifier created")
        except Exception as e:
            print(f"   ❌ Error creating hard voting: {e}")
        
        # 2. Voting Classifier (Soft Voting)
        try:
            voting_soft = VotingClassifier(
                estimators=base_estimators,
                voting='soft'
            )
            voting_soft.fit(X_train_scaled, y_train)
            self.ensemble_models['voting_soft'] = voting_soft
            print("   ✅ Soft Voting Classifier created")
        except Exception as e:
            print(f"   ❌ Error creating soft voting: {e}")
        
        # 3. Stacking Classifier
        try:
            # Use logistic regression as meta-learner
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            stacking = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=5,
                stack_method='predict_proba'
            )
            stacking.fit(X_train_scaled, y_train)
            self.ensemble_models['stacking'] = stacking
            print("   ✅ Stacking Classifier created")
        except Exception as e:
            print(f"   ❌ Error creating stacking: {e}")
        
        # 4. Bagging with best performing model
        try:
            # Find best performing model
            best_model_name = max(self.cv_scores.keys(), 
                                key=lambda k: np.mean(self.cv_scores[k]) if isinstance(self.cv_scores[k], (list, np.ndarray)) else self.cv_scores[k])
            best_model = self.models[best_model_name]
            
            bagging = BaggingClassifier(
                base_estimator=best_model,
                n_estimators=10,
                random_state=42,
                n_jobs=-1
            )
            bagging.fit(X_train_scaled, y_train)
            self.ensemble_models['bagging'] = bagging
            print(f"   ✅ Bagging Classifier created (base: {best_model_name})")
        except Exception as e:
            print(f"   ❌ Error creating bagging: {e}")
        
        # 5. AdaBoost with Decision Tree
        try:
            ada_boost = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=50,
                learning_rate=1.0,
                random_state=42
            )
            ada_boost.fit(X_train_scaled, y_train)
            self.ensemble_models['adaboost'] = ada_boost
            print("   ✅ AdaBoost Classifier created")
        except Exception as e:
            print(f"   ❌ Error creating AdaBoost: {e}")
        
        # Evaluate ensemble models
        if X_val is not None and y_val is not None:
            print("\n📊 Evaluating Ensemble Models:")
            for name, model in self.ensemble_models.items():
                try:
                    y_pred = model.predict(X_val_scaled)
                    accuracy = accuracy_score(y_val, y_pred)
                    print(f"   {name}: {accuracy:.4f}")
                except Exception as e:
                    print(f"   {name}: Error - {e}")
        
        print(f"\n🎉 Ensemble creation completed! {len(self.ensemble_models)} ensemble models created.")
        return self.ensemble_models
    
    def create_custom_stacking(self, X_train, y_train, X_val=None, y_val=None, 
                             meta_learners=None, cv_folds=5):
        """
        Create custom stacking ensemble with multiple meta-learners.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            meta_learners (list): List of meta-learners to try
            cv_folds (int): Number of CV folds for stacking
            
        Returns:
            dict: Dictionary of custom stacking models
        """
        if meta_learners is None:
            meta_learners = [
                ('logistic', LogisticRegression(random_state=42, max_iter=1000)),
                ('ridge', RidgeClassifier(random_state=42)),
                ('svm', SVC(random_state=42, probability=True)),
                ('rf', RandomForestClassifier(random_state=42, n_estimators=50))
            ]
        
        print("\n🔧 CREATING CUSTOM STACKING MODELS")
        print("-" * 40)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Get base estimators
        base_estimators = [(name, model) for name, model in self.models.items() 
                          if hasattr(model, 'predict_proba')]
        
        if len(base_estimators) < 2:
            print("⚠️ Need at least 2 base models for stacking")
            return {}
        
        custom_stacking_models = {}
        
        for meta_name, meta_learner in meta_learners:
            try:
                stacking = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=meta_learner,
                    cv=cv_folds,
                    stack_method='predict_proba',
                    n_jobs=-1
                )
                stacking.fit(X_train_scaled, y_train)
                custom_stacking_models[f'stacking_{meta_name}'] = stacking
                print(f"   ✅ Stacking with {meta_name} meta-learner created")
                
                # Evaluate if validation data provided
                if X_val is not None and y_val is not None:
                    y_pred = stacking.predict(X_val_scaled)
                    accuracy = accuracy_score(y_val, y_pred)
                    print(f"      Validation accuracy: {accuracy:.4f}")
                    
            except Exception as e:
                print(f"   ❌ Error creating stacking with {meta_name}: {e}")
        
        self.ensemble_models.update(custom_stacking_models)
        return custom_stacking_models
    
    def hyperparameter_optimization(self, X_train, y_train, model_name, 
                                  method='grid', n_iter=50, cv=5):
        """
        Advanced hyperparameter optimization using GridSearch or RandomizedSearch.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            model_name (str): Name of the model to optimize
            method (str): Optimization method ('grid' or 'random')
            n_iter (int): Number of iterations for RandomizedSearch
            cv (int): Number of CV folds
            
        Returns:
            object: Best model with optimized parameters
        """
        print(f"\n🎯 HYPERPARAMETER OPTIMIZATION ({method.upper()})")
        print(f"Model: {model_name}")
        print("-" * 40)
        
        configs = self.get_model_configs()
        if model_name not in configs:
            print(f"❌ Model '{model_name}' not found")
            return None
        
        config = configs[model_name]
        model = config['model']
        param_grid = config['params']
        
        # Scale features if needed
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Choose optimization method
        if method.lower() == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='accuracy', 
                n_jobs=-1, verbose=0
            )
        elif method.lower() == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, 
                scoring='accuracy', n_jobs=-1, verbose=0, random_state=42
            )
        else:
            print(f"❌ Unknown optimization method: {method}")
            return None
        
        # Perform search
        try:
            search.fit(X_train_scaled, y_train)
            
            # Store results
            self.best_params[model_name] = search.best_params_
            self.models[model_name] = search.best_estimator_
            
            # Cross-validation score
            cv_scores = cross_val_score(search.best_estimator_, X_train_scaled, y_train, cv=cv)
            self.cv_scores[model_name] = cv_scores
            
            print(f"   ✅ Best parameters: {search.best_params_}")
            print(f"   ✅ Best CV score: {search.best_score_:.4f}")
            print(f"   ✅ CV std: {cv_scores.std():.4f}")
            
            return search.best_estimator_
            
        except Exception as e:
            print(f"   ❌ Error during optimization: {e}")
            return None
    
    def get_model_explanations(self):
        """
        Get detailed explanations of all available models.
        
        Returns:
            dict: Model explanations
        """
        configs = self.get_model_configs()
        explanations = {}
        
        for model_name, config in configs.items():
            if 'description' in config:
                explanations[model_name] = config['description']
        
        return explanations
    
    def create_model_selection_guide(self):
        """
        Create a comprehensive model selection guide.
        
        Returns:
            str: Model selection guide
        """
        guide = """
        Model Selection Guide:

        - Use **Logistic Regression** for interpretable, fast, and linear problems.
        - Use **Random Forest** for robust, high-accuracy results on complex or mixed-type data.
        - Use **Gradient Boosting/XGBoost/LightGBM/CatBoost** for high accuracy on structured/tabular data, especially when non-linear relationships are present.
        - Use **SVM** for small-to-medium datasets with clear margins of separation.
        - Use **KNN** for simple, small datasets where interpretability is not a concern.
        - Use **Naive Bayes** for text classification or when features are independent.
        - Use **Decision Tree** for interpretable models and when you want to visualize decision rules.
        - Use **Ensemble Methods** (Voting, Stacking, Bagging, AdaBoost) to combine strengths of multiple models for improved performance.
        - Consider **hyperparameter tuning** and **feature importance** analysis for optimal results.

        Select models based on dataset size, interpretability needs, and performance requirements.
        """
        return guide
            
