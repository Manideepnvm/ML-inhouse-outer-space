import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import joblib

# Local imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing.data_processor import AstronomicalDataProcessor
from src.models.ml_models import MLModelTrainer
from src.models.deep_learning import ImageClassifier
from src.visualization.visualizer import EnhancedAstronomicalVisualizer
from src import config


st.set_page_config(page_title="Astronomy Data Dashboard", layout="wide")

# Comprehensive color mapping for different object types
# Comprehensive color mapping for different object types
OBJECT_COLORS = config.OBJECT_COLORS
OBJECT_DESCRIPTIONS = config.OBJECT_DESCRIPTIONS


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


@st.cache_data(show_spinner=False)
def load_and_process_files(csv_paths_tuple: tuple):
    """
    Load and process multiple CSV files, concatenate them with a source_file column.
    Accepts a tuple of file paths (tuples are hashable so caching works).
    Returns concatenated raw, cleaned, engineered dataframes and feature summary based on concatenated engineered df.
    """
    processor = AstronomicalDataProcessor()
    dfs = []
    for p in csv_paths_tuple:
        d = processor.load_data(p)
        if d is None:
            continue
        # keep track of source file
        d['source_file'] = os.path.basename(p)
        dfs.append(d)

    if not dfs:
        return None, None, None, None

    # Concatenate raw data (union of columns)
    data_concat = pd.concat(dfs, ignore_index=True, sort=False)

    # Process concatenated data
    data_clean = processor.clean_data(data_concat)
    data_eng = processor.engineer_features(data_clean)
    feature_summary = processor.create_feature_summary(data_eng)
    return data_concat, data_clean, data_eng, feature_summary


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

    # Discover CSV files in data/ folder
    csv_paths = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.csv")))
    if not csv_paths:
        st.error(f"No CSV files found in {config.DATA_DIR} folder.")
        st.info(f"Please add your dataset CSV files into the {config.DATA_DIR} directory.")
        return

    # Sidebar: dataset selector (All concatenated or a single file)
    basenames = [os.path.basename(p) for p in csv_paths]
    dataset_options = ["All (concatenated)"] + basenames
    selected = st.sidebar.selectbox("Dataset to view", options=dataset_options, index=0)

    with st.spinner("Loading and processing data..."):
        if selected == "All (concatenated)":
            data, data_clean, data_eng, feature_summary = load_and_process_files(tuple(csv_paths))
            used_files = basenames
        else:
            # find the full path for the selected basename
            sel_path = next((p for p in csv_paths if os.path.basename(p) == selected), None)
            data, data_clean, data_eng, feature_summary = load_and_process(sel_path)
            used_files = [selected] if data is not None else []

    if data is None:
        st.error("Failed to load data.")
        return

    st.success(f"Loaded: {len(data):,} rows, {data.shape[1]} columns. After engineering: {data_eng.shape[1]} features.")
    st.info(f"Files used: {', '.join(used_files)}")

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
        " 🧪 Testing analysis",
        " 🖼️ Image Testing Analysis",
    ])

    # Initialize components for testing
    processor = AstronomicalDataProcessor()
    ml_trainer = MLModelTrainer()

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
        
        st.subheader("📊 Raw Data Sample")
        st.markdown("First 50 rows of the original dataset:")
        st.dataframe(data.head(50))
        
        st.subheader("⚙️ Engineered Data Sample")
        st.markdown("First 50 rows after feature engineering (new features added):")
        st.dataframe(data_eng.head(50))

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

    # Testing Section
    with tabs[8]:
        st.subheader("🧪 Single Object Prediction")
        st.markdown("""
        Test the model with your own data. Enter the astronomical features below to get a classification prediction.
        The model will predict whether the object is a **STAR**, **GALAXY**, or **QSO** (Quasar).
        """)
        
        # Create form for input
        with st.form("prediction_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ra = st.number_input("Right Ascension (ra)", value=185.0, format="%.6f")
                dec = st.number_input("Declination (dec)", value=0.0, format="%.6f")
            
            with col2:
                u = st.number_input("u (ultraviolet)", value=19.0, format="%.6f")
                g = st.number_input("g (green)", value=18.0, format="%.6f")
            
            with col3:
                r = st.number_input("r (red)", value=17.0, format="%.6f")
                i = st.number_input("i (near infrared)", value=17.0, format="%.6f")
            
            with col4:
                z = st.number_input("z (infrared)", value=17.0, format="%.6f")
                redshift = st.number_input("Redshift", value=0.0001, format="%.8f")
            
            submitted = st.form_submit_button("🔍 Predict Class")
            
        if submitted:
            if data is None:
                st.error("Please load the dataset first (from the sidebar) to initialize the model pipeline.")
            else:
                with st.spinner("Processing data and generating prediction..."):
                    try:
                        # 1. Create input dataframe
                        input_data = pd.DataFrame([{
                            'ra': ra, 'dec': dec, 
                            'u': u, 'g': g, 'r': r, 'i': i, 'z': z, 
                            'redshift': redshift,
                            'class': 'STAR' # Dummy target for pipeline compatibility
                        }])
                        
                        # 2. Process Input FIRST to determine available features
                        input_clean = processor.clean_data(input_data)
                        input_eng = processor.engineer_features(input_clean)
                        
                        # Identify the features available from the input form
                        # (excluding dummy class and non-numeric)
                        feature_cols_input = input_eng.select_dtypes(include=[np.number]).columns.tolist()
                        if 'class' in feature_cols_input: feature_cols_input.remove('class')
                        
                        # 3. Re-run complete pipeline on FULL data to ensure scaler/selector consistency
                        # This is expensive but necessary since pipeline state wasn't persisted
                        # Optimally, we would pickle the fitted processor, but here we must refit.
                        
                        # Use cached processed data if possible, or process fresh
                        # Use cached processed data if possible, or process fresh
                        if 'processor_state_v2' not in st.session_state:
                            # OPTIMIZATION: Try to load pre-trained pipeline first
                            if processor.load_pipeline(config.PROCESSOR_STATE_PATH):
                                st.success("✅ Loaded pre-trained pipeline state!")
                                st.session_state['processor_state_v2'] = processor
                            else:
                                st.warning("⚠️ Pre-trained pipeline not found. Falling back to slow training...")
                                st.info("Initializing feature pipeline (training on full dataset)...")
                                # Process full dataset to fit scaler
                                # We use the loaded 'data' from dashboard
                                
                                # Filter to ensure target exists (concatenated data might have NaNs for target)
                                t_col = 'class' # default
                                if target_col in data.columns:
                                    t_col = target_col
                                    
                                training_data_subset = data.copy()
                                if t_col in training_data_subset.columns:
                                    initial_len = len(training_data_subset)
                                    training_data_subset = training_data_subset.dropna(subset=[t_col])
                                    if len(training_data_subset) < initial_len:
                                        st.info(f"Filtered {initial_len - len(training_data_subset)} rows with missing target '{t_col}' for training pipeline.")
                                
                                if len(training_data_subset) == 0:
                                    st.error(f"No valid training data found with target '{t_col}'. cannot initialize pipeline.")
                                    st.stop()
    
                                training_data_clean = processor.clean_data(training_data_subset)
                                training_data_eng = processor.engineer_features(training_data_clean)
                                
                                # Find target
                                t_col = 'class' # default
                                if target_col in training_data_eng.columns:
                                    t_col = target_col
                                
                                # STRICT ALIGNMENT: Only keep features that exist in input
                                available_training_cols = [c for c in feature_cols_input if c in training_data_eng.columns]
                                
                                # Add target back for prepare_features
                                cols_to_keep = available_training_cols + [t_col]
                                training_data_eng = training_data_eng[cols_to_keep]
                                
                                X_train_full, y_train_full = processor.prepare_features(training_data_eng, target_col=t_col)
                                
                                # Fit Scaler
                                processor.scale_features(X_train_full, method='standard')
                                
                                # Fit Selector
                                processor.select_features(processor.scaler.transform(X_train_full), y_train_full, method='mutual_info', k=20)
                                
                                st.session_state['processor_state_v2'] = processor
                                # Save for next time
                                processor.save_pipeline(config.PROCESSOR_STATE_PATH)
                        else:
                            processor = st.session_state['processor_state_v2']

                        # 4. Prepare matched input
                        X_input = input_eng[feature_cols_input].copy()
                        
                        # Align input columns to what scaler obtained (should be exact match now)
                        # processor.scaler.feature_names_in_ if available
                        
                        # To be safe, we rely on the processor's fitted scaler
                        try:
                            X_input_scaled = processor.scaler.transform(X_input)
                        except ValueError as e:
                            st.warning(f"Feature mismatch: {e}. Attempting to align columns...")
                            # Fallback: recreate scaler on input just to proceed (Not ideal but prevents crash)
                            X_input_scaled = processor.scaler.fit_transform(X_input)
                            pass
                            
                        # Transform with selector
                        X_input_selected = processor.feature_selector.transform(X_input_scaled)
                        
                        # 5. Load Model
                        # Try loading Random Forest as default best
                        model_path = os.path.join(config.MODELS_DIR, "random_forest_model.joblib")
                        if not os.path.exists(model_path):
                            st.error(f"Model file not found at {model_path}")
                        else:
                            model = joblib.load(model_path)
                            
                            # Predict
                            prediction_code = model.predict(X_input_selected)[0]
                            try:
                                prediction_proba = model.predict_proba(X_input_selected)[0]
                            except:
                                prediction_proba = None
                            
                            # Decode class
                            # We need the class map. unique classes from data.
                            # 'classes' are in alphabetic order usually for LabelEncoder
                            # Or we can check processor.label_encoder.classes_
                            classes = processor.label_encoder.classes_
                            predicted_class = classes[prediction_code]
                            
                            # Display Result
                            st.markdown("### 🎯 Prediction Result")
                            
                            color = get_object_color(predicted_class)
                            st.markdown(f"""
                            <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                                <h2 style="color: {color}; margin:0;">{predicted_class}</h2>
                                <p>{OBJECT_DESCRIPTIONS.get(predicted_class, '')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if prediction_proba is not None:
                                st.markdown("#### Confidence Scores")
                                # Create a nice bar chart for probabilities
                                prob_df = pd.DataFrame({
                                    'Class': classes,
                                    'Probability': prediction_proba
                                })
                                
                                fig_probs = px.bar(
                                    prob_df, x='Probability', y='Class', orientation='h',
                                    title="Prediction Percentages",
                                    color='Class',
                                    color_discrete_map=OBJECT_COLORS,
                                    text=prob_df['Probability'].apply(lambda x: f"{x:.1%}")
                                )
                                fig_probs.update_layout(height=300)
                                st.plotly_chart(fig_probs, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        st.exception(e)

    # Image Analysis
    with tabs[9]:
        st.subheader("🖼️ Image Testing Analysis")
        st.markdown("Upload an image of an astronomical object to classify it as **STAR**, **GALAXY**, or **QSO**.")
        
        # Load Model Metrics
        metrics = None
        metrics_path = config.IMAGE_METRICS_PATH
        if os.path.exists(metrics_path):
            try:
                import json
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            except Exception as e:
                st.error(f"Error loading model metrics: {e}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            img_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
            
            if img_file is not None:
                st.image(img_file, caption="Uploaded Image", use_column_width=True)
                
        with col2:
            if img_file is not None:
                st.info("Ready to analyze.")
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image features with CNN..."):
                        # Initialize on demand
                        if 'image_classifier' not in st.session_state:
                            st.session_state['image_classifier'] = ImageClassifier()
                        
                        classifier = st.session_state['image_classifier']
                        
                        # Load image from the uploaded file buffer as PIL
                        from PIL import Image
                        image = Image.open(img_file)
                        
                        results = classifier.predict_image(image)
                        
                        if results:
                            # Display results
                            best_class = max(results, key=results.get)
                            prob = results[best_class]
                            
                            color = OBJECT_COLORS.get(best_class, '#808080')
                            
                            # Prediction Card
                            st.markdown(f"""
                            <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; margin-bottom: 20px;">
                                <h3 style="color: {color}; margin:0;">Prediction: {best_class}</h3>
                                <p style="font-size: 1.2em;">Confidence: <b>{prob:.1%}</b></p>
                                <p>This image represents a <b>{best_class}</b> based on its visual features.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Chart
                            st.markdown("#### Probability Distribution")
                            probs_df = pd.DataFrame({
                                'Class': list(results.keys()),
                                'Probability': list(results.values())
                            })
                            
                            fig = px.bar(
                                probs_df, x='Probability', y='Class', orientation='h',
                                color='Class', color_discrete_map=OBJECT_COLORS,
                                text=probs_df['Probability'].apply(lambda x: f"{x:.1%}")
                            )
                            fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show specific metrics for this predicted class
                            if metrics:
                                st.divider()
                                st.subheader(f"📊 Analysis Data for {best_class}")
                                
                                report = metrics.get('report', {})
                                class_metrics = report.get(best_class, {})
                                
                                if class_metrics:
                                    # Create a focused table for this specific result
                                    st.info(f"The following data is specific to the **{best_class}** classification:")
                                    
                                    specific_data = pd.DataFrame([{
                                        'Metric': 'Prediction Confidence',
                                        'Value': f"{prob:.2%}",
                                        'Description': 'Probability for this specific image'
                                    }, {
                                        'Metric': 'Model Precision', 
                                        'Value': f"{class_metrics.get('precision', 0):.2%}",
                                        'Description': f'Accuracy when model predicts {best_class}'
                                    }, {
                                        'Metric': 'Model Recall',
                                        'Value': f"{class_metrics.get('recall', 0):.2%}",
                                        'Description': f'Ability to find all real {best_class}s'
                                    }, {
                                        'Metric': 'F1-Score',
                                        'Value': f"{class_metrics.get('f1-score', 0):.2%}",
                                        'Description': 'Balance between Precision and Recall'
                                    }])
                                    
                                    st.dataframe(
                                        specific_data,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                    
                                    st.caption(f"Overall Model Accuracy: {metrics.get('test_accuracy', 0):.2%}")

                        else:
                            st.error("Model prediction failed. Please ensure the backend supports this file format.")
            else:
                st.info("👈 Please upload an image to start.")

    st.sidebar.divider()
    st.sidebar.markdown("**Object Type Colors:**")
    st.sidebar.markdown("🟡 Stars (Gold)")
    st.sidebar.markdown("🔵 Galaxies (Blue)")  
    st.sidebar.markdown("🔴 Quasars (Red)")
    st.sidebar.divider()
    st.sidebar.markdown("Run locally: `python -m streamlit run dashboard.py`")


if __name__ == "__main__":
    main()


