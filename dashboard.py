import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local imports
from src.preprocessing.data_processor import AstronomicalDataProcessor
from src.visualization.visualizer import EnhancedAstronomicalVisualizer


st.set_page_config(page_title="Astronomy Data Dashboard", layout="wide")

# Comprehensive color mapping for different object types
OBJECT_COLORS = {
    # Primary object types (as they appear in the dataset)
    'STAR': '#FFD700',      # Gold - represents stellar objects
    'GALAXY': '#4169E1',    # Royal Blue - represents galaxies
    'QSO': '#DC143C',       # Crimson - represents quasars
    
    # Alternative naming conventions
    'Star': '#FFD700',
    'Galaxy': '#4169E1', 
    'Quasar': '#DC143C',
    
    # Numeric encodings (if target is encoded)
    0: '#FFD700',           # Star
    1: '#4169E1',           # Galaxy
    2: '#DC143C',           # Quasar
    
    # Additional astronomical object types
    'GALAXY_ACTIVE': '#8A2BE2',  # Blue Violet for active galaxies
    'STAR_BINARY': '#FFA500',    # Orange for binary stars
    'QSO_BL_LAC': '#FF1493',     # Deep Pink for BL Lac objects
}

# Object type descriptions for better understanding
OBJECT_DESCRIPTIONS = {
    'STAR': 'Stars are luminous celestial bodies that generate energy through nuclear fusion in their cores. They appear as point sources of light.',
    'GALAXY': 'Galaxies are massive collections of stars, gas, dust, and dark matter bound together by gravity. They can contain billions of stars.',
    'QSO': 'Quasars (Quasi-Stellar Objects) are extremely luminous active galactic nuclei powered by supermassive black holes at their centers.'
}


@st.cache_data(show_spinner=False)
def load_and_process(csv_path: str):
    processor = AstronomicalDataProcessor()
    data = processor.load_data(csv_path)
    if data is None:
        return None, None, None, None
    data_clean = processor.clean_data(data)
    data_eng = processor.engineer_features(data_clean)
    feature_summary = processor.create_feature_summary(data_eng)
    return data, data_clean, data_eng, feature_summary


def get_object_color(obj_type):
    """Get color for object type with fallback"""
    return OBJECT_COLORS.get(obj_type, '#808080')  # Gray fallback


def main():
    st.title("🔭 Astronomical Data Dashboard")
    st.caption("Interactive exploration of photometric colors, redshift features, correlations, and more.")
    
    # Add comprehensive information about the dataset
    st.markdown("""

    """)
    
    # Color legend with explanations
    st.markdown("### 🎨 Object Type Color Coding")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
       
        """)
    
    with col2:
        st.markdown("""
      
        """)
    
    with col3:
        st.markdown("""
       
        """)

    # Only read from data folder
    default_path = os.path.join("data", "Skyserver_SQL2_27_2018_6_51_39_PM.csv")
    
    if not os.path.exists(default_path):
        st.error(f"Data file not found: {default_path}")
        st.info("Please ensure your Skyserver dataset is in the data/ folder with the correct filename.")
        return

    with st.spinner("Loading and processing data..."):
        data, data_clean, data_eng, feature_summary = load_and_process(default_path)

    if data is None:
        st.error("Failed to load data.")
        return

    st.success(f"Loaded: {len(data):,} rows, {data.shape[1]} columns. After engineering: {data_eng.shape[1]} features.")

    # Controls
    target_col = st.sidebar.selectbox("Target column", options=[c for c in data.columns if c in ("class", "Class", "target")] + list(data.columns), index=0)

    vis = EnhancedAstronomicalVisualizer()

    tabs = st.tabs([
        " Overview",
        " Color Indices", 
        " Redshift Analysis",
        " Correlations",
        " HR Diagram",
        " Interactive 3D",
        " Sky Map",
        " Feature Summary",
    ])

    # Overview
    with tabs[0]:
        st.subheader("📋 Data Overview")
        st.markdown("""
        This section provides a comprehensive view of the raw and processed astronomical data.
        """)
        
        # Data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Objects", f"{len(data):,}")
        with col2:
            st.metric("Features", len(data.columns))
        with col3:
            st.metric("Engineered Features", len(data_eng.columns))
        with col4:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.subheader("📊 Raw Data Sample")
        st.markdown("First 50 rows of the original dataset:")
        st.dataframe(data.head(50))
        
        st.subheader("⚙️ Engineered Data Sample")
        st.markdown("First 50 rows after feature engineering (new features added):")
        st.dataframe(data_eng.head(50))
        
        # Data quality information
        st.subheader("🔍 Data Quality Assessment")
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            st.warning(f"⚠️ Missing values detected: {missing_data.sum()} total missing values")
            st.dataframe(missing_data[missing_data > 0].to_frame('Missing Count'))
        else:
            st.success("✅ No missing values in the dataset")

    # Color Indices
    with tabs[1]:
        st.subheader("🌈 Color Indices and Color-Color Diagrams")
        st.markdown("""
       
        """)
        
        required = ["g", "r", "i"]
        if all(col in data_eng.columns for col in required):
            g_r = data_eng["g"] - data_eng["r"]
            r_i = data_eng["r"] - data_eng["i"]
            color_df = pd.DataFrame({"g-r": g_r, "r-i": r_i})
            
            # Color statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("g-r Mean", f"{g_r.mean():.3f}")
            with col2:
                st.metric("r-i Mean", f"{r_i.mean():.3f}")
            with col3:
                st.metric("Color Range", f"{g_r.max() - g_r.min():.3f}")
            
            # Color-color diagram with object types
            if target_col in data_eng.columns:
                st.markdown("### 🎨 Color-Color Diagram")
                st.markdown("""
         
                """)
                
                fig_scatter = px.scatter(
                    data_eng, x=g_r, y=r_i, color=target_col,
                    title="Color-Color Diagram (g-r vs r-i) by Object Type",
                    labels={'x': 'g-r (Color Index)', 'y': 'r-i (Color Index)'},
                    color_discrete_map=OBJECT_COLORS,
                    opacity=0.7
                )
                fig_scatter.update_layout(height=500, showlegend=True)
                fig_scatter.update_traces(marker=dict(size=4))
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Distributions
            st.markdown("### 📊 Color Index Distributions")
            st.markdown("""
            These histograms show the distribution of color indices across all objects. 
            The shape reveals the population characteristics of different object types.
            """)
            
            fig_dist = make_subplots(rows=1, cols=2, subplot_titles=['g-r Distribution', 'r-i Distribution'])
            
            fig_dist.add_trace(go.Histogram(x=g_r, name='g-r', marker_color='#FF6B6B', opacity=0.7), row=1, col=1)
            fig_dist.add_trace(go.Histogram(x=r_i, name='r-i', marker_color='#4ECDC4', opacity=0.7), row=1, col=2)
            
            fig_dist.update_layout(height=400, showlegend=False)
            fig_dist.update_xaxes(title_text="g-r Color Index", row=1, col=1)
            fig_dist.update_xaxes(title_text="r-i Color Index", row=1, col=2)
            fig_dist.update_yaxes(title_text="Count", row=1, col=1)
            fig_dist.update_yaxes(title_text="Count", row=1, col=2)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Color statistics by object type
            if target_col in data_eng.columns:
                st.markdown("### 📈 Color Statistics by Object Type")
                color_stats = data_eng.groupby(target_col)[['g', 'r', 'i']].agg(['mean', 'std']).round(3)
                st.dataframe(color_stats)
        else:
            st.error("❌ Required photometric bands (g, r, i) not found in the dataset.")

    # Redshift
    with tabs[2]:
        st.subheader("🔴 Redshift Analysis")
        st.markdown("""
     
        """)
        
        if "redshift" in data_eng.columns:
            redshift = data_eng["redshift"]
            
            # Redshift statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Objects with Redshift", f"{(redshift > 0).sum():,}")
            with col2:
                st.metric("Max Redshift", f"{redshift.max():.3f}")
            with col3:
                st.metric("Mean Redshift", f"{redshift[redshift > 0].mean():.3f}")
            with col4:
                st.metric("Zero Redshift", f"{(redshift == 0).sum():,}")
            
            # Redshift distribution by object type
            if target_col in data_eng.columns:
                redshift_pos = data_eng[redshift > 0]
                if len(redshift_pos) > 0:
                    st.markdown("### 📊 Redshift Distribution by Object Type")
                    st.markdown("""
                   
                    """)
                    
                    fig_redshift = px.histogram(
                        redshift_pos, x=np.log10(redshift_pos["redshift"]), 
                        color=target_col, nbins=60,
                        title="log₁₀(Redshift) Distribution by Object Type",
                        color_discrete_map=OBJECT_COLORS,
                        opacity=0.7
                    )
                    fig_redshift.update_layout(
                        height=500,
                        xaxis_title="log₁₀(Redshift)",
                        yaxis_title="Count",
                        showlegend=True
                    )
                    st.plotly_chart(fig_redshift, use_container_width=True)
                    
                    # Redshift statistics by object type
                    st.markdown("### 📈 Redshift Statistics by Object Type")
                    redshift_stats = redshift_pos.groupby(target_col)['redshift'].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
                    st.dataframe(redshift_stats)
            else:
                # Simple histogram if no target column
                st.markdown("### 📊 Overall Redshift Distribution")
                fig_redshift = px.histogram(
                    data_eng[redshift > 0], x=np.log10(data_eng.loc[redshift > 0, "redshift"]), 
                    nbins=60, title="log₁₀(Redshift) Distribution",
                    color_discrete_sequence=['#3498DB']
                )
                fig_redshift.update_layout(height=500)
                st.plotly_chart(fig_redshift, use_container_width=True)
            
            # Category counts
            if "redshift_category" in data_eng.columns:
                st.markdown("### 🏷️ Redshift Categories")
                st.markdown("""

                """)
                
                cat_counts = data_eng["redshift_category"].value_counts().reset_index()
                cat_counts.columns = ["category", "count"]
                fig_cat = px.bar(cat_counts, x="category", y="count", 
                               title="Redshift Categories Distribution",
                               color="count", color_continuous_scale="viridis")
                fig_cat.update_layout(height=400)
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.error("❌ Redshift column not found in the dataset.")

    # Correlations
    with tabs[3]:
        st.subheader("🔗 Feature Correlations")
        st.markdown("""
        
        """)
        
        num_df = data_eng.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            corr = num_df.corr()
            
            # Correlation statistics
            st.markdown("### 📊 Correlation Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Features Analyzed", len(corr.columns))
            with col2:
                st.metric("Strong Correlations (|r| > 0.7)", (abs(corr) > 0.7).sum().sum() - len(corr))
            with col3:
                st.metric("Mean Correlation", f"{corr.abs().mean().mean():.3f}")
            
            # Correlation matrix
            st.markdown("### 🔥 Correlation Heatmap")
            st.markdown("""
            The heatmap shows Pearson correlation coefficients between all numeric features:
            - **Red**: Strong positive correlation (r > 0.5)
            - **Blue**: Strong negative correlation (r < -0.5)
            - **White**: Weak correlation (|r| < 0.3)
            """)
            
            fig_corr = px.imshow(corr, color_continuous_scale="RdBu", origin="lower", 
                               title="Feature Correlation Heatmap", aspect="auto")
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Top correlations
            st.markdown("### 🎯 Top Feature Correlations")
            # Get upper triangle of correlation matrix
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            corr_upper = corr.where(mask)
            
            # Flatten and sort correlations
            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_val = corr.iloc[i, j]
                    if not np.isnan(corr_val):
                        corr_pairs.append({
                            'Feature 1': corr.columns[i],
                            'Feature 2': corr.columns[j],
                            'Correlation': corr_val,
                            'Abs Correlation': abs(corr_val)
                        })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Abs Correlation', ascending=False)
            st.dataframe(corr_df.head(10).round(3))
            
        else:
            st.error("❌ Not enough numeric features for correlation analysis.")

    # HR Diagram
    with tabs[4]:
        st.subheader("⭐ Hertzsprung-Russell (H-R) Diagram")
        st.markdown("""
        
        """)
        
        if all(c in data_eng.columns for c in ["g", "r"]):
            df = data_eng[["g", "r"]].copy()
            df["g-r"] = df["g"] - df["r"]
            
            # Data validation
            valid_data = df.dropna()
            if len(valid_data) == 0:
                st.error("❌ No valid data points for H-R diagram (all values are NaN)")
            else:
                st.markdown(f"### 📊 H-R Diagram Analysis ({len(valid_data):,} valid points)")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("g-r Range", f"{df['g-r'].min():.2f} to {df['g-r'].max():.2f}")
                with col2:
                    st.metric("r Magnitude Range", f"{df['r'].min():.2f} to {df['r'].max():.2f}")
                with col3:
                    st.metric("Data Completeness", f"{len(valid_data)/len(df)*100:.1f}%")
                
                if target_col in data_eng.columns:
                    df[target_col] = data_eng[target_col]
                    st.markdown("""
                 
                    """)
                    
                    fig = px.scatter(df, x="g-r", y="r", color=target_col, opacity=0.6, 
                                   title="Hertzsprung-Russell Diagram (g-r vs r) by Object Type",
                                   labels={'g-r': 'Color Index (g-r)', 'r': 'r-band Magnitude'},
                                   color_discrete_map=OBJECT_COLORS)
                else:
                    fig = px.scatter(df, x="g-r", y="r", opacity=0.6, 
                                   title="Hertzsprung-Russell Diagram (g-r vs r)",
                                   labels={'g-r': 'Color Index (g-r)', 'r': 'r-band Magnitude'})
                
                fig.update_yaxes(autorange="reversed")  # Brighter objects at top
                fig.update_layout(
                    height=600,
                    xaxis_title="Color Index (g-r)",
                    yaxis_title="r-band Magnitude (brighter → fainter)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional analysis
                if target_col in data_eng.columns:
                    st.markdown("### 📈 H-R Diagram Statistics by Object Type")
                    hr_stats = df.groupby(target_col)[['g-r', 'r']].agg(['mean', 'std', 'min', 'max']).round(3)
                    st.dataframe(hr_stats)
        else:
            st.error("❌ Required photometric bands (g, r) not found in the dataset.")

    # Interactive 3D
    with tabs[5]:
        st.subheader("🌌 Interactive 3D Visualization")
        st.markdown("""

        """)
        
        possible_axes = [c for c in ["u", "g", "r", "i", "z", "ra", "dec", "redshift"] if c in data_eng.columns]
        if len(possible_axes) >= 3:
            st.markdown("### 🎛️ Visualization Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X axis", options=possible_axes, index=0, 
                                   help="Choose the feature for the X-axis")
                y_col = st.selectbox("Y axis", options=possible_axes, index=min(1, len(possible_axes)-1),
                                   help="Choose the feature for the Y-axis")
            
            with col2:
                z_col = st.selectbox("Z axis", options=possible_axes, index=min(2, len(possible_axes)-1),
                                   help="Choose the feature for the Z-axis")
                color_col = st.selectbox("Color by", options=[target_col] + possible_axes, index=0,
                                       help="Choose the feature for color coding")
            
            # Data validation
            valid_data = data_eng[[x_col, y_col, z_col]].dropna()
            if len(valid_data) == 0:
                st.error("❌ No valid data points for 3D visualization (all values are NaN)")
            else:
                st.markdown(f"### 📊 3D Visualization ({len(valid_data):,} valid points)")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{x_col} Range", f"{data_eng[x_col].min():.2f} to {data_eng[x_col].max():.2f}")
                with col2:
                    st.metric(f"{y_col} Range", f"{data_eng[y_col].min():.2f} to {data_eng[y_col].max():.2f}")
                with col3:
                    st.metric(f"{z_col} Range", f"{data_eng[z_col].min():.2f} to {data_eng[z_col].max():.2f}")
                
                fig3d = vis.create_interactive_3d_visualization(data_eng, x_col, y_col, z_col, color_col=color_col)
                st.plotly_chart(fig3d, use_container_width=True)
                
                # Instructions
                st.markdown("""
            
                """)
        else:
            st.error("❌ Need at least three numeric columns among u,g,r,i,z,ra,dec,redshift for 3D visualization.")

    # Sky Map
    with tabs[6]:
        st.subheader("🗺️ Sky Coordinates Map")
        st.markdown("""
        
        """)
        
        if all(c in data_eng.columns for c in ["ra", "dec"]):
            # Data validation
            valid_coords = data_eng[["ra", "dec"]].dropna()
            if len(valid_coords) == 0:
                st.error("❌ No valid coordinate data for sky map (all values are NaN)")
            else:
                st.markdown(f"### 🌍 Sky Distribution Analysis ({len(valid_coords):,} valid points)")
                
                # Coordinate statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RA Range", f"{data_eng['ra'].min():.1f}° to {data_eng['ra'].max():.1f}°")
                with col2:
                    st.metric("DEC Range", f"{data_eng['dec'].min():.1f}° to {data_eng['dec'].max():.1f}°")
                with col3:
                    st.metric("Sky Coverage", f"{(data_eng['ra'].max() - data_eng['ra'].min()):.1f}° × {(data_eng['dec'].max() - data_eng['dec'].min()):.1f}°")
                with col4:
                    st.metric("Data Completeness", f"{len(valid_coords)/len(data_eng)*100:.1f}%")
                
                if target_col in data_eng.columns:
                    st.markdown("### 🎨 Sky Map by Object Type")
                    st.markdown("""
                  
                    """)
                    
                    fig_geo = px.scatter_geo(
                        data_eng, lat="dec", lon="ra", color=target_col,
                        title="Sky Distribution (RA/DEC) by Object Type",
                        color_discrete_map=OBJECT_COLORS,
                        opacity=0.7
                    )
                else:
                    fig_geo = px.scatter_geo(
                        data_eng, lat="dec", lon="ra",
                        title="Sky Distribution (RA/DEC)",
                        opacity=0.7
                    )
                
                fig_geo.update_traces(marker=dict(size=3, opacity=0.6))
                fig_geo.update_layout(
                    height=600,
                    geo=dict(
                        showframe=True,
                        showcoastlines=True,
                        projection_type='equirectangular'
                    )
                )
                st.plotly_chart(fig_geo, use_container_width=True)
                
                # Additional analysis
                if target_col in data_eng.columns:
                    st.markdown("### 📈 Spatial Statistics by Object Type")
                    spatial_stats = data_eng.groupby(target_col)[['ra', 'dec']].agg(['mean', 'std', 'min', 'max']).round(2)
                    st.dataframe(spatial_stats)
        else:
            st.error("❌ Required coordinate columns (ra, dec) not found in the dataset.")

    # Feature summary
    with tabs[7]:
        st.subheader("📊 Feature Summary")
        st.markdown("""
       
        """)
        
        # Feature summary statistics
        st.markdown("### 📈 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", len(feature_summary))
        with col2:
            st.metric("Complete Features", (feature_summary['Missing_Count'] == 0).sum())
        with col3:
            st.metric("Features with Missing Data", (feature_summary['Missing_Count'] > 0).sum())
        with col4:
            st.metric("High Variance Features", (feature_summary['Std'] > feature_summary['Std'].quantile(0.9)).sum())
        
        # Feature summary table
        st.markdown("### 📋 Detailed Feature Statistics")
        st.markdown("""
      
        """)
        
        # Add search and filter functionality
        search_term = st.text_input("🔍 Search features:", placeholder="Type to filter features...")
        if search_term:
            filtered_summary = feature_summary[feature_summary['Feature'].str.contains(search_term, case=False, na=False)]
            st.dataframe(filtered_summary, use_container_width=True)
        else:
            st.dataframe(feature_summary, use_container_width=True)
        
        # Feature quality assessment
        st.markdown("### 🔍 Feature Quality Assessment")
        
        # Missing data analysis
        missing_features = feature_summary[feature_summary['Missing_Count'] > 0]
        if len(missing_features) > 0:
            st.warning(f"⚠️ {len(missing_features)} features have missing data")
            st.dataframe(missing_features[['Feature', 'Missing_Count', 'Missing_Percentage']].round(2))
        else:
            st.success("✅ No missing data in any features")
        
        # High variance features
        high_var_features = feature_summary[feature_summary['Std'] > feature_summary['Std'].quantile(0.9)]
        if len(high_var_features) > 0:
            st.info(f"📊 Top {len(high_var_features)} highest variance features:")
            st.dataframe(high_var_features[['Feature', 'Std', 'Min', 'Max']].round(3))
        
        # Low variance features (potential constants)
        low_var_features = feature_summary[feature_summary['Std'] < 0.01]
        if len(low_var_features) > 0:
            st.warning(f"⚠️ {len(low_var_features)} features have very low variance (potential constants):")
            st.dataframe(low_var_features[['Feature', 'Std', 'Min', 'Max']].round(6))

    st.sidebar.divider()
    st.sidebar.markdown("**Object Type Colors:**")
    st.sidebar.markdown("🟡 Stars (Gold)")
    st.sidebar.markdown("🔵 Galaxies (Blue)")  
    st.sidebar.markdown("🔴 Quasars (Red)")
    st.sidebar.divider()
    st.sidebar.markdown("Run locally: `python -m streamlit run dashboard.py`")


if __name__ == "__main__":
    main()


