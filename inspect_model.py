
import joblib
import os
import pandas as pd

def inspect():
    model_path = 'models/random_forest_model.joblib'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    
    if hasattr(model, 'feature_names_in_'):
        print("Feature names found:")
        print(model.feature_names_in_)
    elif hasattr(model, 'n_features_in_'):
        print(f"Number of features expected: {model.n_features_in_}")
    else:
        print("Could not determine feature info.")

    # Check feature importance file
    fi_path = 'models/feature_importance.joblib'
    if os.path.exists(fi_path):
        fi = joblib.load(fi_path)
        print("\nFeature Importance Keys/Data:")
        print(fi.head() if isinstance(fi, pd.DataFrame) else fi)

if __name__ == '__main__':
    inspect()
