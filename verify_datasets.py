
import pandas as pd
import glob
import os

def check_datasets():
    required_features = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']
    target_col = 'class'
    
    csv_files = glob.glob('data/*.csv')
    
    results = []
    
    for file in csv_files:
        print(f"\nChecking {file}...")
        try:
            # Read only header and a few rows
            df = pd.read_csv(file, nrows=5)
            cols = [c.lower() for c in df.columns]
            
            # Check availability
            missing = [f for f in required_features if f not in cols and f != target_col]
            
            # Target might be named differently
            target_found = False
            for t in ['class', 'target', 'label']:
                if t in cols:
                    target_found = t
                    break
            
            status = "Compatible"
            if missing:
                status = f"Missing: {missing}"
            if not target_found:
                status += f", Target Missing"
                
            print(f"  Status: {status}")
            
            if status == "Compatible":
                # Extract one of each class
                df_full = pd.read_csv(file)
                # Normalize columns for consistency
                df_full.columns = [c.lower() for c in df_full.columns]
                
                # Get samples
                samples = df_full.groupby(target_found).head(1)
                
                # Keep only relevant columns for display
                display_cols = required_features + [target_found]
                samples = samples[display_cols]
                
                # Rename target to class for consistency
                if target_found != 'class':
                    samples = samples.rename(columns={target_found: 'class'})
                    
                results.append((os.path.basename(file), samples))
                
        except Exception as e:
            print(f"  Error reading file: {e}")

    # Generate Report
    with open('dataset_verification.md', 'w') as f:
        f.write("# Dataset Verification Report\n\n")
        
        for name, samples in results:
            f.write(f"## {name}\n")
            f.write("Status: ✅ Compatible\n\n")
            f.write(samples.to_markdown(index=False))
            f.write("\n\n")
            
    print("\nReport generated: dataset_verification.md")

if __name__ == "__main__":
    check_datasets()
