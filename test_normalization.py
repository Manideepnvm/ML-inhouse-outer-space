
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.preprocessing.data_processor import AstronomicalDataProcessor

def test_class_normalization():
    print("🧪 Testing Class Normalization...")
    
    # Create dummy data with mixed class names
    data = pd.DataFrame({
        'ra': [1, 2, 3, 4, 5, 6, 7],
        'dec': [1, 2, 3, 4, 5, 6, 7],
        'class': [
            'stars',      # lowercase
            'Galaxy',     # Mixed case
            'quasers',    # Misspelled
            'STAR',       # Correct
            'qso',        # Lowercase acronym
            'GAL',        # Abbreviation
            'quasar'      # Variation
        ],
        'u': [1,1,1,1,1,1,1], # Numeric cols to avoid dropping
        'g': [1,1,1,1,1,1,1],
        'r': [1,1,1,1,1,1,1],
        'i': [1,1,1,1,1,1,1],
        'z': [1,1,1,1,1,1,1],
        'redshift': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    })
    
    print("Original Classes:", data['class'].tolist())
    
    processor = AstronomicalDataProcessor()
    clean_data = processor.clean_data(data)
    
    result_classes = clean_data['class'].tolist()
    print("Normalized Classes:", result_classes)
    
    expected = ['STAR', 'GALAXY', 'QSO', 'STAR', 'QSO', 'GALAXY', 'QSO']
    
    assert result_classes == expected, f"❌ Validation Failed! Expected {expected}, got {result_classes}"
    print("✅ Validation Passed! All classes normalized correctly.")

if __name__ == "__main__":
    test_class_normalization()
