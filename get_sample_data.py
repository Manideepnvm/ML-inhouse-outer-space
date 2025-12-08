
import pandas as pd
import os

def get_data():
    try:
        df = pd.read_csv('data/SDSS_DR18.csv')
        cols = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'class']
        
        # Get one of each
        samples = df[cols].groupby('class').head(1)
        
        # Format as markdown
        md = samples.to_markdown(index=False)
        
        with open('sample_data.md', 'w') as f:
            f.write(md)
            
        print("Data written to sample_data.md")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_data()
