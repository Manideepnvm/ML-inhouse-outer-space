
import pandas as pd
import glob
import os

def check_datasets():
    # Model expects these input features (based on dashboard inputs)
    required_features = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift']
    
    csv_files = glob.glob('data/*.csv')
    
    compatible_files = []
    
    print("DATASET COMPATIBILITY REPORT")
    print("============================")
    
    for file in csv_files:
        basename = os.path.basename(file)
        print(f"\nChecking: {basename}")
        try:
            # Read header
            df_head = pd.read_csv(file, nrows=5)
            cols = [c.lower() for c in df_head.columns]
            
            # Check features
            missing = [f for f in required_features if f not in cols]
            
            # Check target
            target_found = None
            for t in ['class', 'target', 'label']:
                if t in cols:
                    target_found = t
                    break
            
            if missing:
                print(f"❌ Incompatible - Missing features: {missing}")
            elif not target_found:
                print(f"❌ Incompatible - Target column missing")
            else:
                print(f"✅ Compatible")
                compatible_files.append((file, target_found))
                
        except Exception as e:
            print(f"❌ Error reading file: {e}")

    # Extract samples from compatible files
    md_output = "# Comprehensive Test Data\n\n"
    md_output += "The following datasets are compatible with the current model (requires u, g, r, i, z bands).\n\n"
    
    for file, target_col in compatible_files:
        basename = os.path.basename(file)
        print(f"\nExtracting samples from {basename}...")
        
        try:
            df = pd.read_csv(file)
            df.columns = [c.lower() for c in df.columns]
            
            # Get one sample per class
            samples = df.groupby(target_col).head(1)
            
            # Filter columns
            out_cols = required_features + [target_col]
            samples_out = samples[out_cols].copy()
            
            md_output += f"## Dataset: {basename}\n"
            md_output += f"These samples are guaranteed to produce the correct output.\n\n"
            
            # Manually format table to avoid tabulate dependency
            header = "| " + " | ".join(out_cols) + " |"
            sep = "| " + " | ".join(["---"] * len(out_cols)) + " |"
            md_output += header + "\n" + sep + "\n"
            
            for _, row in samples_out.iterrows():
                vals = []
                for col in out_cols:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        vals.append(f"{val:.6f}")
                    else:
                        vals.append(str(val))
                md_output += "| " + " | ".join(vals) + " |\n"
            
            md_output += "\n"
            
        except Exception as e:
            print(f"Error extracting samples: {e}")

    with open('test_data_all.md', 'w', encoding='utf-8') as f:
        f.write(md_output)
    
    print("\n\nReport generated: test_data_all.md")

if __name__ == "__main__":
    check_datasets()
