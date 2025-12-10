import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data Files
MAIN_DATA_FILENAME = 'Skyserver_SQL2_27_2018_6_51_39_PM.csv'
MAIN_DATA_PATH = os.path.join(DATA_DIR, MAIN_DATA_FILENAME)
PROCESSOR_STATE_PATH = os.path.join(MODELS_DIR, 'processor_state.joblib')
EVALUATION_REPORT_PATH = os.path.join(RESULTS_DIR, 'evaluation_report.txt')
IMAGE_METRICS_PATH = os.path.join(RESULTS_DIR, 'image_metrics.json')

# Column configuration
TARGET_COL = 'class'

# Object Mappings
OBJECT_COLORS = {
    # Primary object types
    'STAR': '#FFD700',      # Gold
    'GALAXY': '#4169E1',    # Royal Blue
    'QSO': '#DC143C',       # Crimson
    
    # Numeric
    0: '#FFD700',
    1: '#4169E1',
    2: '#DC143C',
}

OBJECT_DESCRIPTIONS = {
    'STAR': 'Stars are luminous celestial bodies that generate energy through nuclear fusion in their cores. They appear as point sources of light.',
    'GALAXY': 'Galaxies are massive collections of stars, gas, dust, and dark matter bound together by gravity. They can contain billions of stars.',
    'QSO': 'Quasars (Quasi-Stellar Objects) are extremely luminous active galactic nuclei powered by supermassive black holes at their centers.'
}

# Image Data Configuration
IMAGE_CLASS_MAPPING = {
    'stars': 'STAR', 
    'Galaxy': 'GALAXY', 
    'quasers': 'QSO' # Handling legacy folder name
}

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 3,
    'deep_learning_epochs': 20,
    'deep_learning_batch_size': 64,
    'max_features': 20,
    'quick_mode': True
}
