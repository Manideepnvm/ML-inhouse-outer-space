"""
Astronomical Data Processor
Handles data loading, cleaning, feature engineering, and preprocessing for astronomical object classification.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class AstronomicalDataProcessor:
    """
    A comprehensive data processor for astronomical datasets.
    """
    
    def __init__(self):
        """Initialize the data processor with default configurations."""
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.imputer = None
        
    def load_data(self, file_path):
        """
        Load astronomical data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data or None if failed
        """
        try:
            print(f"📂 Loading data from: {file_path}")
            data = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully!")
            print(f"   Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def explore_data(self, data):
        """
        Perform comprehensive data exploration.
        
        Args:
            data (pd.DataFrame): Input data
        """
        print("\n🔍 DATA EXPLORATION")
        print("-" * 40)
        
        # Basic info
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Data types
        print("\n📊 Data Types:")
        print(data.dtypes.value_counts())
        
        # Missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠️ Missing Values:")
            missing_df = missing[missing > 0].sort_values(ascending=False)
            print(missing_df)
        else:
            print("\n✅ No missing values found!")
        
        # Class distribution
        if 'class' in data.columns:
            print("\n📈 Class Distribution:")
            class_counts = data['class'].value_counts()
            print(class_counts)
            print(f"Class proportions:\n{data['class'].value_counts(normalize=True).round(3)}")
        
        # Statistical summary
        print("\n📊 Statistical Summary:")
        print(data.describe())
        
        # Duplicate rows
        duplicates = data.duplicated().sum()
        print(f"\n🔄 Duplicate rows: {duplicates}")
        
    def clean_data(self, data):
        """
        Clean the astronomical data.
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        print("\n🧹 CLEANING DATA")
        print("-" * 40)
        
        data_clean = data.copy()
        initial_shape = data_clean.shape
        
        # Remove duplicates
        data_clean = data_clean.drop_duplicates()
        print(f"   Removed {initial_shape[0] - data_clean.shape[0]} duplicate rows")
        
        # Handle missing values
        missing_threshold = 0.5  # Remove columns with >50% missing values
        missing_ratio = data_clean.isnull().sum() / len(data_clean)
        cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
        
        if cols_to_drop:
            data_clean = data_clean.drop(columns=cols_to_drop)
            print(f"   Removed columns with >50% missing: {cols_to_drop}")
        
        # Fill remaining missing values
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = data_clean.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with median
        if len(numeric_cols) > 0:
            self.imputer = SimpleImputer(strategy='median')
            data_clean[numeric_cols] = self.imputer.fit_transform(data_clean[numeric_cols])
        
        # Impute categorical columns with mode
        for col in categorical_cols:
            if data_clean[col].isnull().sum() > 0:
                mode_value = data_clean[col].mode()
                if len(mode_value) > 0:
                    data_clean[col].fillna(mode_value[0], inplace=True)
        
        # Remove outliers using IQR method for numerical columns
        for col in numeric_cols:
            if col != 'class':  # Don't remove outliers from target
                Q1 = data_clean[col].quantile(0.25)
                Q3 = data_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = len(data_clean)
                data_clean = data_clean[
                    (data_clean[col] >= lower_bound) & 
                    (data_clean[col] <= upper_bound)
                ]
                outliers_removed = outliers_before - len(data_clean)
                if outliers_removed > 0:
                    print(f"   Removed {outliers_removed} outliers from {col}")
        
        print(f"✅ Data cleaning completed!")
        print(f"   Final shape: {data_clean.shape}")
        print(f"   Rows removed: {initial_shape[0] - data_clean.shape[0]}")
        
        return data_clean
    
    def engineer_features(self, data):
        """
        Engineer new features for astronomical classification.
        
        Args:
            data (pd.DataFrame): Cleaned data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        print("\n⚙️ FEATURE ENGINEERING")
        print("-" * 40)
        
        data_eng = data.copy()
        initial_features = data_eng.shape[1]
        
        # Color indices (differences between magnitudes)
        magnitude_cols = ['u', 'g', 'r', 'i', 'z']
        available_mags = [col for col in magnitude_cols if col in data_eng.columns]
        
        if len(available_mags) >= 2:
            print("   Creating color indices...")
            # Common astronomical color indices
            if 'u' in available_mags and 'g' in available_mags:
                data_eng['u_g'] = data_eng['u'] - data_eng['g']
            if 'g' in available_mags and 'r' in available_mags:
                data_eng['g_r'] = data_eng['g'] - data_eng['r']
            if 'r' in available_mags and 'i' in available_mags:
                data_eng['r_i'] = data_eng['r'] - data_eng['i']
            if 'i' in available_mags and 'z' in available_mags:
                data_eng['i_z'] = data_eng['i'] - data_eng['z']
            if 'g' in available_mags and 'i' in available_mags:
                data_eng['g_i'] = data_eng['g'] - data_eng['i']
        
        # Magnitude ratios
        if len(available_mags) >= 2:
            print("   Creating magnitude ratios...")
            for i, mag1 in enumerate(available_mags):
                for mag2 in available_mags[i+1:]:
                    # Avoid division by zero
                    ratio_col = f'{mag1}_{mag2}_ratio'
                    data_eng[ratio_col] = np.where(
                        data_eng[mag2] != 0,
                        data_eng[mag1] / data_eng[mag2],
                        0
                    )
        
        # Coordinate-based features
        coordinate_cols = ['ra', 'dec']
        available_coords = [col for col in coordinate_cols if col in data_eng.columns]
        
        if len(available_coords) == 2:
            print("   Creating coordinate-based features...")
            # Distance from galactic center (simplified)
            galactic_center_ra = 266.4  # degrees
            galactic_center_dec = -29.0  # degrees
            
            data_eng['distance_from_gc'] = np.sqrt(
                (data_eng['ra'] - galactic_center_ra)**2 + 
                (data_eng['dec'] - galactic_center_dec)**2
            )
            
            # Galactic latitude approximation
            data_eng['abs_galactic_lat'] = np.abs(data_eng['dec'])
        
        # Redshift-based features
        if 'redshift' in data_eng.columns:
            print("   Creating redshift-based features...")
            # Log redshift (handle zeros)
            data_eng['log_redshift'] = np.log1p(np.abs(data_eng['redshift']))
            
            # Redshift categories
            data_eng['redshift_category'] = pd.cut(
                data_eng['redshift'], 
                bins=[-np.inf, 0.1, 0.5, 1.0, np.inf],
                labels=['very_low', 'low', 'medium', 'high']
            )
        
        # Photometric features
        if len(available_mags) >= 3:
            print("   Creating photometric features...")
            # Total flux (sum of all magnitudes)
            mag_subset = data_eng[available_mags]
            data_eng['total_magnitude'] = mag_subset.sum(axis=1)
            data_eng['mean_magnitude'] = mag_subset.mean(axis=1)
            data_eng['magnitude_std'] = mag_subset.std(axis=1)
            
            # Brightness measure (inverse of magnitude)
            for mag in available_mags:
                flux_col = f'{mag}_flux'
                data_eng[flux_col] = 10**(-0.4 * data_eng[mag])
        
        # Error-based features
        error_cols = [col for col in data_eng.columns if 'err' in col.lower()]
        if error_cols:
            print("   Creating error-based features...")
            # Signal-to-noise ratio
            for err_col in error_cols:
                base_col = err_col.replace('err', '').replace('_', '')
                if base_col in data_eng.columns:
                    snr_col = f'{base_col}_snr'
                    data_eng[snr_col] = np.where(
                        data_eng[err_col] != 0,
                        data_eng[base_col] / data_eng[err_col],
                        0
                    )
        
        # Polynomial features for key variables
        if 'redshift' in data_eng.columns:
            print("   Creating polynomial features...")
            data_eng['redshift_squared'] = data_eng['redshift']**2
            data_eng['redshift_cubed'] = data_eng['redshift']**3
        
        # Encode categorical features
        categorical_cols = data_eng.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop('class', errors='ignore')  # Don't encode target
        
        if len(categorical_cols) > 0:
            print(f"   Encoding categorical features: {list(categorical_cols)}")
            for col in categorical_cols:
                le = LabelEncoder()
                data_eng[f'{col}_encoded'] = le.fit_transform(data_eng[col].astype(str))
        
        # Handle infinite values
        data_eng = data_eng.replace([np.inf, -np.inf], np.nan)
        
        # Fill any new NaN values created during feature engineering
        numeric_cols = data_eng.select_dtypes(include=[np.number]).columns
        data_eng[numeric_cols] = data_eng[numeric_cols].fillna(data_eng[numeric_cols].median())
        
        final_features = data_eng.shape[1]
        print(f"✅ Feature engineering completed!")
        print(f"   Initial features: {initial_features}")
        print(f"   Final features: {final_features}")
        print(f"   New features created: {final_features - initial_features}")
        
        return data_eng
    
    def prepare_features(self, data, target_col='class'):
        """
        Prepare features and target for machine learning.
        
        Args:
            data (pd.DataFrame): Processed data
            target_col (str): Target column name
            
        Returns:
            tuple: (X, y) features and target
        """
        print(f"\n📋 PREPARING FEATURES")
        print("-" * 40)
        
        # Separate features and target
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(data[target_col])
        
        # Select feature columns (exclude target and non-numeric)
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        
        X = data[feature_cols].copy()
        
        # Handle any remaining missing values
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Feature columns: {len(feature_cols)}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        print(f"   Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def scale_features(self, X, method='standard'):
        """
        Scale features using specified method.
        
        Args:
            X (pd.DataFrame): Features to scale
            method (str): Scaling method ('standard', 'minmax')
            
        Returns:
            np.ndarray: Scaled features
        """
        print(f"\n⚖️ SCALING FEATURES ({method})")
        print("-" * 40)
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Features scaled using {method} scaling")
        print(f"   Original shape: {X.shape}")
        print(f"   Scaled shape: {X_scaled.shape}")
        
        return X_scaled
    
    def select_features(self, X, y, method='mutual_info', k=50):
        """
        Select the most important features.
        
        Args:
            X (np.ndarray): Scaled features
            y (np.ndarray): Target variable
            method (str): Feature selection method
            k (int): Number of features to select
            
        Returns:
            np.ndarray: Selected features
        """
        print(f"\n🎯 FEATURE SELECTION ({method}, k={k})")
        print("-" * 40)
        
        # Ensure k doesn't exceed available features
        k = min(k, X.shape[1])
        
        if method == 'mutual_info':
            score_func = mutual_info_classif
        elif method == 'f_classif':
            score_func = f_classif
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get feature scores
        scores = self.feature_selector.scores_
        selected_indices = self.feature_selector.get_support(indices=True)
        
        print(f"✅ Feature selection completed!")
        print(f"   Original features: {X.shape[1]}")
        print(f"   Selected features: {X_selected.shape[1]}")
        print(f"   Top 10 feature scores: {sorted(scores, reverse=True)[:10]}")
        
        return X_selected
    
    def handle_class_imbalance(self, X, y, method='smote'):
        """
        Handle class imbalance using specified method.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            method (str): Resampling method
            
        Returns:
            tuple: Resampled (X, y)
        """
        print(f"\n⚖️ HANDLING CLASS IMBALANCE ({method})")
        print("-" * 40)
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"   Original distribution: {dict(zip(unique, counts))}")
        
        try:
            if method == 'smote':
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=42)
            elif method == 'adasyn':
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(random_state=42)
            elif method == 'random_oversample':
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=42)
            elif method == 'random_undersample':
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=42)
            else:
                print(f"⚠️ Unknown method '{method}', skipping resampling")
                return X, y
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Check new distribution
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            print(f"   New distribution: {dict(zip(unique_new, counts_new))}")
            print(f"✅ Class imbalance handled!")
            
            return X_resampled, y_resampled
            
        except ImportError:
            print("⚠️ imbalanced-learn not installed, skipping resampling")
            return X, y
        except Exception as e:
            print(f"⚠️ Error in resampling: {e}, skipping")
            return X, y
    
    def create_feature_summary(self, data):
        """
        Create a comprehensive feature summary.
        
        Args:
            data (pd.DataFrame): Processed data
            
        Returns:
            pd.DataFrame: Feature summary
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        summary = pd.DataFrame({
            'Feature': numeric_cols,
            'Type': 'Numeric',
            'Missing_Count': data[numeric_cols].isnull().sum().values,
            'Missing_Percentage': (data[numeric_cols].isnull().sum() / len(data) * 100).values,
            'Mean': data[numeric_cols].mean().values,
            'Std': data[numeric_cols].std().values,
            'Min': data[numeric_cols].min().values,
            'Max': data[numeric_cols].max().values,
            'Unique_Values': data[numeric_cols].nunique().values
        })
        
        return summary.round(4)
    
    def save_processed_data(self, data, file_path):
        """
        Save processed data to file.
        
        Args:
            data (pd.DataFrame): Processed data
            file_path (str): Output file path
        """
        try:
            data.to_csv(file_path, index=False)
            print(f"💾 Processed data saved to: {file_path}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def load_processed_data(self, file_path):
        """
        Load previously processed data.
        
        Args:
            file_path (str): Path to processed data file
            
        Returns:
            pd.DataFrame: Loaded processed data
        """
        try:
            data = pd.read_csv(file_path)
            print(f"✅ Processed data loaded from: {file_path}")
            return data
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return None