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

# Color mapping for different object types
OBJECT_COLORS = {
    'STAR': '#FFD700',      # Gold
    'GALAXY': '#4169E1',    # Royal Blue  
    'QSO': '#DC143C',       # Crimson
    'Star': '#FFD700',
    'Galaxy': '#4169E1', 
    'Quasar': '#DC143C',
    0: '#FFD700',           # Star
    1: '#4169E1',           # Galaxy
    2: '#DC143C'            # Quasar
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
        "Overview",
        "Color Indices",
        "Redshift",
        "Correlations",
        "HR Diagram",
        "Interactive 3D",
        "Sky Map",
        "Feature Summary",
    ])

    # Overview
    with tabs[0]:
        st.subheader("Data Overview")
        st.dataframe(data.head(50))
        st.subheader("Engineered Data Sample")
        st.dataframe(data_eng.head(50))

    # Color Indices
    with tabs[1]:
        st.subheader("Color Indices and Color-Color Diagrams")
        required = ["g", "r", "i"]
        if all(col in data_eng.columns for col in required):
            g_r = data_eng["g"] - data_eng["r"]
            r_i = data_eng["r"] - data_eng["i"]
            color_df = pd.DataFrame({"g-r": g_r, "r-i": r_i})
            
            # Color-color diagram with object types
            if target_col in data_eng.columns:
                fig_scatter = px.scatter(
                    data_eng, x=g_r, y=r_i, color=target_col,
                    title="Color-Color Diagram (g-r vs r-i)",
                    labels={'x': 'g-r', 'y': 'r-i'},
                    color_discrete_map=OBJECT_COLORS
                )
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Distributions
            st.write("Color Index Distributions")
            fig_dist = make_subplots(rows=1, cols=2, subplot_titles=['g-r Distribution', 'r-i Distribution'])
            
            fig_dist.add_trace(go.Histogram(x=g_r, name='g-r', marker_color='#FF6B6B'), row=1, col=1)
            fig_dist.add_trace(go.Histogram(x=r_i, name='r-i', marker_color='#4ECDC4'), row=1, col=2)
            
            fig_dist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Need bands g, r, i for color-color plots.")

    # Redshift
    with tabs[2]:
        st.subheader("Redshift-based Features")
        if "redshift" in data_eng.columns:
            redshift = data_eng["redshift"]
            st.metric("Non-zero redshift count", int((redshift > 0).sum()))
            
            # Redshift distribution by object type
            if target_col in data_eng.columns:
                redshift_pos = data_eng[redshift > 0]
                if len(redshift_pos) > 0:
                    fig_redshift = px.histogram(
                        redshift_pos, x=np.log10(redshift_pos["redshift"]), 
                        color=target_col, nbins=60,
                        title="log10(redshift) distribution by Object Type",
                        color_discrete_map=OBJECT_COLORS
                    )
                    fig_redshift.update_layout(height=500)
                    st.plotly_chart(fig_redshift, use_container_width=True)
            else:
                # Simple histogram if no target column
                fig_redshift = px.histogram(
                    data_eng[redshift > 0], x=np.log10(data_eng.loc[redshift > 0, "redshift"]), 
                    nbins=60, title="log10(redshift) distribution"
                )
                st.plotly_chart(fig_redshift, use_container_width=True)
            
            # Category counts
            if "redshift_category" in data_eng.columns:
                cat_counts = data_eng["redshift_category"].value_counts().reset_index()
                cat_counts.columns = ["category", "count"]
                fig_cat = px.bar(cat_counts, x="category", y="count", title="Redshift Categories")
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Redshift column not found.")

    # Correlations
    with tabs[3]:
        st.subheader("Correlation Matrix (numeric features)")
        num_df = data_eng.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            corr = num_df.corr()
            st.dataframe(corr.round(3))
            fig_corr = px.imshow(corr, color_continuous_scale="RdBu", origin="lower", 
                               title="Correlation Heatmap", aspect="auto")
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric features for correlation analysis.")

    # HR Diagram
    with tabs[4]:
        st.subheader("Color-Magnitude (H-R like) Diagram")
        if all(c in data_eng.columns for c in ["g", "r"]):
            df = data_eng[["g", "r"]].copy()
            df["g-r"] = df["g"] - df["r"]
            
            if target_col in data_eng.columns:
                df[target_col] = data_eng[target_col]
                fig = px.scatter(df, x="g-r", y="r", color=target_col, opacity=0.6, 
                               title="Color-Magnitude Diagram (g-r vs r) by Object Type",
                               color_discrete_map=OBJECT_COLORS)
            else:
                fig = px.scatter(df, x="g-r", y="r", opacity=0.6, 
                               title="Color-Magnitude Diagram (g-r vs r)")
            
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need columns g and r.")

    # Interactive 3D
    with tabs[5]:
        st.subheader("Interactive 3D Visualization")
        possible_axes = [c for c in ["u", "g", "r", "i", "z", "ra", "dec", "redshift"] if c in data_eng.columns]
        if len(possible_axes) >= 3:
            x_col = st.selectbox("X axis", options=possible_axes, index=0)
            y_col = st.selectbox("Y axis", options=possible_axes, index=min(1, len(possible_axes)-1))
            z_col = st.selectbox("Z axis", options=possible_axes, index=min(2, len(possible_axes)-1))
            color_col = st.selectbox("Color by", options=[target_col] + possible_axes, index=0)
            fig3d = vis.create_interactive_3d_visualization(data_eng, x_col, y_col, z_col, color_col=color_col)
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Need at least three numeric columns among u,g,r,i,z,ra,dec,redshift.")

    # Sky Map
    with tabs[6]:
        st.subheader("Sky Coordinates Map")
        if all(c in data_eng.columns for c in ["ra", "dec"]):
            if target_col in data_eng.columns:
                # Color by object type
                fig_geo = px.scatter_geo(
                    data_eng, lat="dec", lon="ra", color=target_col,
                    title="Sky Distribution (RA/DEC) by Object Type",
                    color_discrete_map=OBJECT_COLORS
                )
            else:
                fig_geo = px.scatter_geo(
                    data_eng, lat="dec", lon="ra",
                    title="Sky Distribution (RA/DEC)"
                )
            
            fig_geo.update_traces(marker=dict(size=3, opacity=0.6))
            fig_geo.update_layout(height=600)
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("Columns ra and dec not found.")

    # Feature summary
    with tabs[7]:
        st.subheader("Feature Summary (Numeric)")
        st.dataframe(feature_summary)

    st.sidebar.divider()
    st.sidebar.markdown("**Object Type Colors:**")
    st.sidebar.markdown("🟡 Stars (Gold)")
    st.sidebar.markdown("🔵 Galaxies (Blue)")  
    st.sidebar.markdown("🔴 Quasars (Red)")
    st.sidebar.divider()
    st.sidebar.markdown("Run locally: `streamlit run dashboard.py`")


if __name__ == "__main__":
    main()


