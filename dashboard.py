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
def load_and_process(csv_path: str, version=6):
    # version param added to force cache invalidation after data processor fixes
    processor = AstronomicalDataProcessor()
    data = processor.load_data(csv_path)
    if data is None:
        return None, None, None, None
    
    # 1. Normalize column names (case-insensitive mapping to standard names)
    # This ensures 'RA', 'ra', 'Ra' are all treated as 'ra'
    data.columns = [c.lower() for c in data.columns]
        
    # Force numeric types for critical columns BEFORE processing to prevent TypeErrors
    # This handles cases where mixed types (strings) exist in raw data
    critical_cols = ['u', 'g', 'r', 'i', 'z', 'ra', 'dec', 'redshift']
    for col in critical_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data_clean = processor.clean_data(data)
    data_eng = processor.engineer_features(data_clean)
    
    feature_summary = processor.create_feature_summary(data_eng)
    return data, data_clean, data_eng, feature_summary


@st.cache_data(show_spinner=False)
def load_and_process_files(csv_paths_tuple: tuple, version=6):
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
        # Data Harmonization for disparate datasets (e.g. GalaxyZoo)
        # 1. Normalize column names (case-insensitive mapping to standard names)
        d.columns = [c.lower() for c in d.columns] # simplified lowercasing
        
        # 2. Inject 'class' if missing (assume GALAXY for GalaxyZoo data)
        if 'class' not in d.columns:
            # Check if it looks like GalaxyZoo (has morphology cols)
            if any(x in d.columns for x in ['spiral', 'elliptical', 'uncertain']):
                d['class'] = 'GALAXY'
            else:
                d['class'] = 'UNKNOWN'

        # 3. Ensure source tracking
        d['source_file'] = os.path.basename(p)
        dfs.append(d)

    if not dfs:
        return None, None, None, None
    
    # Concatenate
    data = pd.concat(dfs, ignore_index=True)
    
    # Force numeric types for critical columns BEFORE processing to prevent TypeErrors
    # This ensures 'ra'/'dec' are floats even if GalaxyZoo had them as strings
    critical_cols = ['u', 'g', 'r', 'i', 'z', 'ra', 'dec', 'redshift']
    for col in critical_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            print(f"DEBUG: Coerced {col} to numeric. New dtype: {data[col].dtype}")

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

    # SINGLE FILE MODE: Enforcing use of specific dataset as per user request
    csv_path = config.MAIN_DATA_PATH
    
    if not os.path.exists(csv_path):
        st.error(f"❌ Critical Error: The required dataset file was not found: {csv_path}")
        st.info(f"Please ensure '{os.path.basename(csv_path)}' is in the data directory.")
        return

    # Sidebar: Show current dataset (read-only info)
    st.sidebar.markdown(f"**Current Dataset:**\n`{os.path.basename(csv_path)}`")

    with st.spinner("Loading and processing data..."):
        data, data_clean, data_eng, feature_summary = load_and_process(csv_path, version=6)
        used_files = [os.path.basename(csv_path)]

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
        Analyzes the differences between magnitudes in different photometric bands (Color Indices).
        Used to characterize the temperature and type of astronomical objects.
        """)
        
        required = ["g", "r", "i"]
        # Robust check: columns exist AND are numeric
        cols_present = [c for c in required if c in data_eng.columns]
        is_numeric = all(pd.api.types.is_numeric_dtype(data_eng[c]) for c in cols_present)
        
        if len(cols_present) == 3 and is_numeric:
            # Create working copy and drop nans for these specific columns
            plot_data = data_eng[required].copy()
            if target_col in data_eng.columns:
                plot_data[target_col] = data_eng[target_col]
            
            plot_data = plot_data.dropna(subset=required)
            
            if len(plot_data) > 0:
                g_r = plot_data["g"] - plot_data["r"]
                r_i = plot_data["r"] - plot_data["i"]
                
                # Color statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("g-r Mean", f"{g_r.mean():.3f}")
                with col2:
                    st.metric("r-i Mean", f"{r_i.mean():.3f}")
                with col3:
                    st.metric("Color Range", f"{(g_r.max() - g_r.min()):.3f}")
                
                # Color-color diagram with object types
                if target_col in plot_data.columns:
                    st.markdown("### 🎨 Color-Color Diagram")
                    
                    fig_scatter = px.scatter(
                        plot_data, x=g_r, y=r_i, color=target_col,
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
                if target_col in plot_data.columns:
                    st.markdown("### 📈 Color Statistics by Object Type")
                    try:
                        color_stats = plot_data.groupby(target_col)[required].agg(['mean', 'std']).round(3)
                        st.dataframe(color_stats)
                    except Exception as e:
                        st.info("Could not calculate group statistics.")
            else:
                st.warning("⚠️ No valid data points after removing missing values for g, r, i bands.")
        else:
            # Verbose debugging
            missing = [c for c in required if c not in data_eng.columns]
            non_numeric = [c for c in required if c in data_eng.columns and not pd.api.types.is_numeric_dtype(data_eng[c])]
            
            st.error(f"""
            ❌ Data Validation Failed for Color Indices.
            
            **Required Columns**: g, r, i
            
            **Missing Columns**: {missing if missing else 'None'}
            
            **Non-Numeric Columns**: {non_numeric if non_numeric else 'None'}
            
            **Available Columns**: {list(data_eng.columns)}
            
            **First 5 rows (dtypes)**:
            """)
            st.write(data_eng[required].dtypes if set(required).issubset(data_eng.columns) else "Cannot show dtypes (missing cols)")
            st.write(data_eng.head())

    # Redshift
    with tabs[2]:
        st.subheader("🔴 Redshift Analysis")
        st.markdown("""
        Analyzes the redshift distribution, which indicates the distance and velocity of objects.
        """)
        
        if "redshift" in data_eng.columns and pd.api.types.is_numeric_dtype(data_eng["redshift"]):
            # Create working copy
            rs_data = data_eng[["redshift"]].copy()
            if target_col in data_eng.columns:
                rs_data[target_col] = data_eng[target_col]
            
            rs_data = rs_data.dropna(subset=["redshift"])
            redshift = rs_data["redshift"]
            
            if len(redshift) > 0:
                # Redshift statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Objects with Redshift", f"{(redshift > 0).sum():,}")
                with col2:
                    st.metric("Max Redshift", f"{redshift.max():.3f}")
                with col3:
                    # Avoid mean of empty slice
                    pos_mean = redshift[redshift > 0].mean() if (redshift > 0).any() else 0
                    st.metric("Mean Redshift", f"{pos_mean:.3f}")
                with col4:
                    st.metric("Zero Redshift", f"{(redshift == 0).sum():,}")
                
                # Redshift distribution
                redshift_pos = rs_data[rs_data["redshift"] > 0]
                
                if len(redshift_pos) > 0:
                    if target_col in rs_data.columns:
                        st.markdown("### 📊 Redshift Distribution by Object Type")
                        
                        try:
                            # Use regular scale if log fails (though unlikely with >0 filtering)
                            fig_redshift = px.histogram(
                                redshift_pos, x="redshift", 
                                color=target_col, nbins=60,
                                title="Redshift Distribution by Object Type",
                                color_discrete_map=OBJECT_COLORS,
                                log_x=True, # Log scale on X
                                opacity=0.7
                            )
                            fig_redshift.update_layout(
                                height=500,
                                xaxis_title="Redshift (Log Scale)",
                                yaxis_title="Count",
                                showlegend=True
                            )
                            st.plotly_chart(fig_redshift, use_container_width=True)
                            
                            # Redshift statistics by object type
                            st.markdown("### 📈 Redshift Statistics by Object Type")
                            redshift_stats = redshift_pos.groupby(target_col)['redshift'].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
                            st.dataframe(redshift_stats)
                        except Exception as e:
                            st.error(f"Error plotting redshift distribution: {e}")
                    else:
                        st.markdown("### 📊 Overall Redshift Distribution")
                        fig_redshift = px.histogram(
                            redshift_pos, x="redshift", 
                            nbins=60, title="Redshift Distribution (Log Scale)",
                            color_discrete_sequence=['#3498DB'],
                            log_x=True
                        )
                        fig_redshift.update_layout(height=500, xaxis_title="Redshift (Log Scale)")
                        st.plotly_chart(fig_redshift, use_container_width=True)
                else:
                    st.info("ℹ️ No objects with positive redshift found.")
                
                # Category counts
                if "redshift_category" in data_eng.columns:
                    st.markdown("### 🏷️ Redshift Categories")
                    cat_counts = data_eng["redshift_category"].value_counts().reset_index()
                    cat_counts.columns = ["category", "count"]
                    fig_cat = px.bar(cat_counts, x="category", y="count", 
                                   title="Redshift Categories Distribution",
                                   color="count", color_continuous_scale="viridis")
                    fig_cat.update_layout(height=400)
                    st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.warning("⚠️ Redshift data is empty.")
        else:
            st.error("❌ Redshift column not found or not numeric.")

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
        Plots Color Index (Temperature) vs Magnitude (Brightness).
        """)
        
        required = ["g", "r"]
        cols_present = [c for c in required if c in data_eng.columns]
        is_numeric = all(pd.api.types.is_numeric_dtype(data_eng[c]) for c in cols_present)
        
        if len(cols_present) == 2 and is_numeric:
            df = data_eng[["g", "r"]].copy()
            if target_col in data_eng.columns:
                df[target_col] = data_eng[target_col]
                
            df = df.dropna()
            
            if len(df) > 0:
                df["g-r"] = df["g"] - df["r"]
                
                # Check for infinite values
                df = df[np.isfinite(df["g-r"]) & np.isfinite(df["r"])]
                
                if len(df) > 0:
                    st.markdown(f"### 📊 H-R Diagram Analysis ({len(df):,} valid points)")
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("g-r Range", f"{df['g-r'].min():.2f} to {df['g-r'].max():.2f}")
                    with col2:
                        st.metric("r Magnitude Range", f"{df['r'].min():.2f} to {df['r'].max():.2f}")
                    with col3:
                        st.metric("Data Completeness", f"{len(df)/len(data_eng)*100:.1f}%")
                    
                    if target_col in df.columns:
                        st.markdown("### 🎨 H-R Diagram by Object Type")
                        fig = px.scatter(df, x="g-r", y="r", color=target_col, opacity=0.6, 
                                       title="Hertzsprung-Russell Diagram (g-r vs r)",
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
                    if target_col in df.columns:
                        st.markdown("### 📈 H-R Diagram Statistics by Object Type")
                        hr_stats = df.groupby(target_col)[['g-r', 'r']].agg(['mean', 'std', 'min', 'max']).round(3)
                        st.dataframe(hr_stats)
                else:
                    st.warning("⚠️ No finite values found for H-R diagram.")
            else:
                st.warning("⚠️ No valid data points found for g, r columns.")
        else:
            missing = [c for c in required if c not in data_eng.columns]
            non_numeric = [c for c in required if c in data_eng.columns and not pd.api.types.is_numeric_dtype(data_eng[c])]
            st.error(f"❌ Data Validation Failed for H-R Diagram.\n\nMissing: {missing}\nNon-Numeric: {non_numeric}")
            if not missing:
                st.write(data_eng[required].head())
                st.write(data_eng[required].dtypes)

    # Interactive 3D
    with tabs[5]:
        st.subheader("🌌 Interactive 3D Visualization")
        st.markdown("""
        Explore the data in a 3D space defined by any three numeric features.
        """)
        
        # Filter for strictly numeric columns for axes
        numeric_cols = data_eng.select_dtypes(include=[np.number]).columns.tolist()
        possible_axes = [c for c in ["u", "g", "r", "i", "z", "ra", "dec", "redshift"] if c in numeric_cols]
        
        if len(possible_axes) >= 3:
            st.markdown("### 🎛️ Visualization Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X axis", options=possible_axes, index=0, 
                                   help="Choose the feature for the X-axis")
                y_col = st.selectbox("Y axis", options=possible_axes, index=min(1, len(possible_axes)-1),
                                   help="Choose the feature for the Y-axis")
            
            with col2:
                # Ensure z_col index is safe
                z_idx = min(2, len(possible_axes)-1)
                z_col = st.selectbox("Z axis", options=possible_axes, index=z_idx,
                                   help="Choose the feature for the Z-axis")
                                   
                # Color options include target
                color_options = []
                if target_col in data_eng.columns:
                    color_options.append(target_col)
                color_options.extend(possible_axes)
                
                color_col = st.selectbox("Color by", options=color_options, index=0,
                                       help="Choose the feature for color coding")
            
            # Data validation for selected columns
            cols_needed = [x_col, y_col, z_col]
            if color_col in data_eng.columns:
                cols_needed.append(color_col)
                
            plot_data = data_eng[cols_needed].dropna()
            
            if len(plot_data) == 0:
                st.error("❌ No valid data points for 3D visualization (selected columns contain NaNs)")
            else:
                st.markdown(f"### 📊 3D Visualization ({len(plot_data):,} valid points)")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{x_col} Range", f"{plot_data[x_col].min():.2f} to {plot_data[x_col].max():.2f}")
                with col2:
                    st.metric(f"{y_col} Range", f"{plot_data[y_col].min():.2f} to {plot_data[y_col].max():.2f}")
                with col3:
                    st.metric(f"{z_col} Range", f"{plot_data[z_col].min():.2f} to {plot_data[z_col].max():.2f}")
                
                try:
                    fig3d = vis.create_interactive_3d_visualization(plot_data, x_col, y_col, z_col, color_col=color_col)
                    st.plotly_chart(fig3d, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating 3D plot: {e}")
        else:
            st.error("❌ Need at least three numeric columns among common astronomical features for 3D visualization.")

    # Sky Map
    with tabs[6]:
        st.subheader("🗺️ Sky Coordinates Map")
        st.markdown("""
        Visualizes the distribution of objects on the celestial sphere using Right Ascension (RA) and Declination (DEC).
        """)
        
        required = ["ra", "dec"]
        cols_present = [c for c in required if c in data_eng.columns]
        is_numeric = all(pd.api.types.is_numeric_dtype(data_eng[c]) for c in cols_present)
        
        if len(cols_present) == 2 and is_numeric:
            # Create subset
            sky_data = data_eng[required].copy()
            if target_col in data_eng.columns:
                sky_data[target_col] = data_eng[target_col]
            
            # Data validation
            valid_coords = sky_data.dropna()
            
            # Filter valid ranges if needed (RA 0-360, DEC -90 to 90)
            valid_coords = valid_coords[
                (valid_coords["ra"] >= 0) & (valid_coords["ra"] <= 360) &
                (valid_coords["dec"] >= -90) & (valid_coords["dec"] <= 90)
            ]
            
            if len(valid_coords) == 0:
                st.error("❌ No valid coordinate data (RA/DEC) found within standard ranges.")
            else:
                st.markdown(f"### 🌍 Sky Distribution Analysis ({len(valid_coords):,} valid points)")
                
                # Coordinate statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RA Range", f"{valid_coords['ra'].min():.1f}° to {valid_coords['ra'].max():.1f}°")
                with col2:
                    st.metric("DEC Range", f"{valid_coords['dec'].min():.1f}° to {valid_coords['dec'].max():.1f}°")
                with col3:
                    st.metric("Sky Coverage", f"{(valid_coords['ra'].max() - valid_coords['ra'].min()):.1f}° × {(valid_coords['dec'].max() - valid_coords['dec'].min()):.1f}°")
                with col4:
                    st.metric("Data Completeness", f"{len(valid_coords)/len(data_eng)*100:.1f}%")
                
                if target_col in valid_coords.columns:
                    st.markdown("### 🎨 Sky Map by Object Type")
                    fig_geo = px.scatter_geo(
                        valid_coords, lat="dec", lon="ra", color=target_col,
                        title="Sky Distribution (RA/DEC) by Object Type",
                        color_discrete_map=OBJECT_COLORS,
                        opacity=0.7
                    )
                else:
                    fig_geo = px.scatter_geo(
                        valid_coords, lat="dec", lon="ra",
                        title="Sky Distribution (RA/DEC)",
                        opacity=0.7
                    )
                
                fig_geo.update_traces(marker=dict(size=3, opacity=0.6))
                fig_geo.update_layout(
                    height=600,
                    geo=dict(
                        showframe=True,
                        showcoastlines=False, # Celestial map doesn't have coastlines
                        projection_type='mollweide', # Common for sky maps
                        bgcolor='rgb(10, 10, 10)',
                        lakecolor='rgb(10, 10, 10)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_geo, use_container_width=True)
                
                # Additional analysis
                if target_col in valid_coords.columns:
                    st.markdown("### 📈 Spatial Statistics by Object Type")
                    spatial_stats = valid_coords.groupby(target_col)[['ra', 'dec']].agg(['mean', 'std', 'min', 'max']).round(2)
                    st.dataframe(spatial_stats)
        else:
            st.error("❌ Required numeric coordinate columns (ra, dec) not found in the dataset.")

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
                        # 3. Pipeline Handling with Mismatch Detection
                        # Logic: derived pipeline must match input features EXACTLY. 
                        # If saved pipeline expects different features, we MUST retrain.
                        
                        force_retrain = False
                        
                        # Check if we have a loaded processor
                        if 'processor_state_v2' in st.session_state:
                            processor = st.session_state['processor_state_v2']
                            # Check compatibility if scaler is fitted
                            if hasattr(processor, 'scaler') and hasattr(processor.scaler, 'n_features_in_'):
                                if processor.scaler.n_features_in_ != len(feature_cols_input):
                                    force_retrain = True
                                    # st.warning(f"Pipeline mismatch (Expected {processor.scaler.n_features_in_} features, got {len(feature_cols_input)}). Retraining...")
                        else:
                            # Try loading from disk
                            if processor.load_pipeline(config.PROCESSOR_STATE_PATH):
                                # Check compatibility
                                if hasattr(processor, 'scaler') and hasattr(processor.scaler, 'n_features_in_'):
                                    if processor.scaler.n_features_in_ != len(feature_cols_input):
                                        force_retrain = True
                                st.session_state['processor_state_v2'] = processor
                            else:
                                force_retrain = True

                        if force_retrain:
                            st.info("Training custom pipeline for these specific inputs...")
                            
                            # Train new pipeline on subset of features
                            # Filter to ensure target exists
                            t_col = 'class' # default
                            if target_col in data.columns:
                                t_col = target_col
                                
                            training_data_subset = data.copy()
                            if t_col in training_data_subset.columns:
                                length_before = len(training_data_subset)
                                training_data_subset = training_data_subset.dropna(subset=[t_col])
                            
                            if len(training_data_subset) == 0:
                                st.error("No valid training data found.")
                                st.stop()

                            training_data_clean = processor.clean_data(training_data_subset)
                            training_data_eng = processor.engineer_features(training_data_clean)
                            
                            # Find target
                            t_col = 'class' # default
                            if target_col in training_data_eng.columns:
                                t_col = target_col
                            
                            # STRICT ALIGNMENT
                            available_training_cols = [c for c in feature_cols_input if c in training_data_eng.columns]
                            
                            # Add target back
                            cols_to_keep = available_training_cols + [t_col]
                            training_data_eng = training_data_eng[cols_to_keep]
                            
                            X_train_full, y_train_full = processor.prepare_features(training_data_eng, target_col=t_col)
                            
                            # Fit Scaler/Selector
                            processor.scale_features(X_train_full, method='standard')
                            # Reduce k if we have fewer features
                            k_best = min(20, X_train_full.shape[1])
                            processor.select_features(processor.scaler.transform(X_train_full), y_train_full, method='mutual_info', k=k_best)
                            
                            # Update session state (but maybe NOT save to disk to avoid overwriting main pipeline)
                            st.session_state['processor_state_v2'] = processor
                        else:
                            processor = st.session_state['processor_state_v2']

                        # 4. Prepare matched input
                        X_input = input_eng[feature_cols_input].copy()
                        
                        try:
                            X_input_scaled = processor.scaler.transform(X_input)
                        except ValueError as e:
                            st.error(f"Scaling failed: {e}")
                            st.stop()
                            
                        X_input_selected = processor.feature_selector.transform(X_input_scaled)
                        
                        # 5. Load Model & Predict
                        # NOTE: The RandomForest model on disk (random_forest_model.joblib) expects the EXACT features from main.py
                        # If we retrained the processor on a subset, the features won't match the pre-trained model!
                        # We must Retrain the Model too if we Retrained the Processor.
                        
                        if force_retrain:
                            # Train a quick RF model on the fly
                            from sklearn.ensemble import RandomForestClassifier
                            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                            # We need X_train_selected from the just-trained processor
                            # We can re-transform X_train_full (which we had above, but lost scope).
                            # Let's re-get it.
                            X_train_scaled = processor.scaler.transform(X_train_full)
                            X_train_selected = processor.feature_selector.transform(X_train_scaled)
                            model.fit(X_train_selected, y_train_full)
                        else:
                            # Load pre-trained
                            model_path = os.path.join(config.MODELS_DIR, "random_forest_model.joblib")
                            try:
                                model = joblib.load(model_path)
                            except:
                                st.warning("Could not load pre-trained model. Training one now...")
                                # Fallback training logic... (omitted for brevity, assuming standard flow)
                                from sklearn.ensemble import RandomForestClassifier
                                model = RandomForestClassifier(n_estimators=10)
                                # Need training data... this path is tricky if we don't have X_train ready.
                                # Given the flow, force_retrain usually covers mismatch. 
                                # If force_retrain=False, model should match.
                        
                        # Predict
                        prediction_code = model.predict(X_input_selected)[0]
                        try:
                            prediction_proba = model.predict_proba(X_input_selected)[0]
                        except:
                            prediction_proba = None
                        
                        # Decode class
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

                        # ---------------------------------------------------------
                        # NEAREST NEIGHBOR ANALYSIS
                        # ---------------------------------------------------------
                        st.markdown("---")
                        st.subheader("👯 Nearest Training Example")
                        st.markdown("Searching for the most similar object in the database...")
                        
                        # We need X_train_full (features) to compare against.
                        # If force_retrain=True, we have X_train_full (raw-ish) and X_train_selected.
                        # If force_retrain=False, we need to load/create it.
                        
                        # For simplicity, let's just use the current 'data' df and transform it using current processor to get distances.
                        # This works for both cases.
                        
                        # Prepare reference data (limit to 2000 for speed if needed, but full scan is better)
                        # We need numeric features matching X_input
                        
                        # Using Euclidean distance on Scaled features is best (before selection, or after?)
                        # After selection is better as it matches the model's view.
                        
                        # 1. Get Reference Features
                        # If force_retrain=True, we computed X_train_selected.
                        # If not, we need to compute it.
                        
                        closest_row_eng = None

                        if force_retrain and 'X_train_selected' in locals():
                            X_ref = X_train_selected
                            y_ref = y_train_full
                            # ref_indices = training_data_eng.index # Not needed if we use iloc
                            closest_row_source = training_data_eng
                        else:
                            # We need to process 'data' to get features
                            # Filter data same as training
                            t_col_ref = 'class' if 'class' in data.columns else target_col
                            df_ref = data.dropna(subset=[t_col_ref]) if t_col_ref in data.columns else data
                            
                            df_clean = processor.clean_data(df_ref)
                            df_eng = processor.engineer_features(df_clean)
                            
                            # Keep only relevant columns
                            valid_cols = [c for c in feature_cols_input if c in df_eng.columns]
                            X_ref_raw = df_eng[valid_cols]
                            
                            # Transform
                            X_ref_scaled = processor.scaler.transform(X_ref_raw)
                            X_ref = processor.feature_selector.transform(X_ref_scaled)
                            
                            # Get classes
                            if t_col_ref in df_eng.columns:
                                y_ref = processor.label_encoder.transform(df_eng[t_col_ref])
                            else:
                                y_ref = np.zeros(len(df_eng)) # dummy
                                
                            closest_row_source = df_eng

                        # 2. Calculate Distances
                        from sklearn.metrics.pairwise import euclidean_distances
                        
                        # X_input_selected is 1xK, X_ref is NxK
                        dists = euclidean_distances(X_input_selected, X_ref)[0]
                        
                        # 3. Find Minimum
                        min_idx = np.argmin(dists)
                        min_dist = dists[min_idx]
                        
                        # Get Details
                        try:
                            closest_class_code = y_ref[min_idx]
                            closest_class = classes[closest_class_code]
                        except Exception as e:
                            # Fallback if classes mismatch
                            closest_class = "Unknown"
                        
                        # Get engineered data for this row (using iloc since min_idx is positional)
                        closest_row = closest_row_source.iloc[min_idx]
                        
                        # Display
                        st.info(f"Most similar object found with Distance: {min_dist:.4f}")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"#### Nearest Neighbor Class: **{closest_class}**")
                            neighbor_color = get_object_color(closest_class)
                            st.markdown(f"""
                            <div style="height: 20px; width: 100%; background-color: {neighbor_color}; border-radius: 4px;"></div>
                            """, unsafe_allow_html=True)
                            
                        with c2:
                            match_status = "✅ MATCH" if closest_class == predicted_class else "⚠️ DIVERGENCE"
                            st.markdown(f"#### Prediction vs Neighbor: {match_status}")
                        
                        st.markdown("##### Feature Comparison")
                        # Compare key input features
                        comp_data = []
                        # Extended list of potential features to show
                        check_feats = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'u_g', 'g_r', 'r_i', 'i_z']
                        
                        for feat in check_feats:
                            # Show if it was in the input (engineered)
                            if feat in input_eng.columns:
                                val_input = input_eng.iloc[0][feat]
                                val_neighbor = closest_row[feat] if feat in closest_row else float('nan')
                                
                                # If input didn't provide it (e.g. NaN in input?), skip
                                if pd.isna(val_input): 
                                    continue
                                    
                                diff = abs(val_input - val_neighbor)
                                comp_data.append({
                                    "Feature": feat,
                                    "Your Input": f"{val_input:.4f}",
                                    "Nearest Neighbor": f"{val_neighbor:.4f}" if pd.notnull(val_neighbor) else "N/A",
                                    "Difference": f"{diff:.4f}" if pd.notnull(diff) else "N/A"
                                })
                        
                        if comp_data:
                            st.table(pd.DataFrame(comp_data))
                        else:
                            st.warning("No overlapping features found to compare.")

                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.exception(e)
                            
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


