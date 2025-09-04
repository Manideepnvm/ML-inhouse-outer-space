import os
import pandas as pd
import numpy as np
import streamlit as st

# Local imports
from src.preprocessing.data_processor import AstronomicalDataProcessor
from src.visualization.visualizer import EnhancedAstronomicalVisualizer


st.set_page_config(page_title="Astronomy Data Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_and_process(csv_path: str):
    processor = AstronomicalDataProcessor()
    data = processor.load_data(csv_path)
    if data is None:
        return None, None, None
    data_clean = processor.clean_data(data)
    data_eng = processor.engineer_features(data_clean)
    feature_summary = processor.create_feature_summary(data_eng)
    return data, data_clean, data_eng, feature_summary


def main():
    st.title("🔭 Astronomical Data Dashboard")
    st.caption("Interactive exploration of photometric colors, redshift features, correlations, and more.")

    default_path = os.path.join("data", "Skyserver_SQL2_27_2018_6_51_39_PM.csv")

    st.sidebar.header("Data Source")
    csv_path = st.sidebar.text_input("CSV path", value=default_path)
    uploaded = st.sidebar.file_uploader("Or upload CSV", type=["csv"]) 
    if uploaded is not None:
        csv_path = None

    if uploaded is None and (not csv_path or not os.path.exists(csv_path)):
        st.warning("Provide a valid CSV path or upload a CSV file from the Skyserver dataset.")
        return

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        # Save a temp copy to reuse processing pipeline uniformly
        tmp_path = "data/_uploaded_temp.csv"
        os.makedirs("data", exist_ok=True)
        df_raw.to_csv(tmp_path, index=False)
        csv_to_use = tmp_path
    else:
        csv_to_use = csv_path

    with st.spinner("Loading and processing data..."):
        data, data_clean, data_eng, feature_summary = load_and_process(csv_to_use)

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
            st.write("Scatter of color indices (g-r vs r-i)")
            st.plotly_chart(
                vis.create_interactive_3d_visualization(
                    pd.DataFrame({"x": g_r, "y": r_i, "z": g_r*0}),
                    x_col="x", y_col="y", z_col="z"
                ),
                use_container_width=True,
            )
            st.write("Distributions")
            st.altair_chart(
                __import__("altair").Chart(color_df.melt(var_name="index", value_name="value")).mark_area(opacity=0.5).encode(
                    x="value:Q", y="count():Q", color="index:N"
                ).properties(height=300),
                use_container_width=True,
            )
        else:
            st.info("Need bands g, r, i for color-color plots.")

    # Redshift
    with tabs[2]:
        st.subheader("Redshift-based Features")
        if "redshift" in data_eng.columns:
            redshift = data_eng["redshift"]
            st.metric("Non-zero redshift count", int((redshift > 0).sum()))
            st.plotly_chart(
                __import__("plotly.express").histogram(
                    data_eng[redshift > 0], x=np.log10(data_eng.loc[redshift > 0, "redshift"]), nbins=60,
                    title="log10(redshift) distribution"
                ),
                use_container_width=True,
            )
            # Category counts
            if "redshift_category" in data_eng.columns:
                cat_counts = data_eng["redshift_category"].value_counts().reset_index()
                cat_counts.columns = ["category", "count"]
                st.bar_chart(cat_counts.set_index("category"))
        else:
            st.info("Redshift column not found.")

    # Correlations
    with tabs[3]:
        st.subheader("Correlation Matrix (numeric features)")
        num_df = data_eng.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            corr = num_df.corr()
            st.dataframe(corr.round(3))
            st.plotly_chart(
                __import__("plotly.express").imshow(corr, color_continuous_scale="RdBu", origin="lower", title="Correlation Heatmap"),
                use_container_width=True,
            )
        else:
            st.info("Not enough numeric features for correlation analysis.")

    # HR Diagram
    with tabs[4]:
        st.subheader("Color-Magnitude (H-R like) Diagram")
        if all(c in data_eng.columns for c in ["g", "r"]):
            df = data_eng[["g", "r"]].copy()
            df["g-r"] = df["g"] - df["r"]
            fig = __import__("plotly.express").scatter(df, x="g-r", y="r", opacity=0.6, title="Color-Magnitude (g-r vs r)")
            fig.update_yaxes(autorange="reversed")
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
            fig_geo = __import__("plotly.graph_objects").Figure(
                __import__("plotly.graph_objects").Scattergeo(
                    lon=data_eng["ra"], lat=data_eng["dec"], mode="markers",
                    marker=dict(size=3, color="#1f77b4", opacity=0.6)
                )
            )
            fig_geo.update_layout(title="Sky Distribution (RA/DEC)")
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("Columns ra and dec not found.")

    # Feature summary
    with tabs[7]:
        st.subheader("Feature Summary (Numeric)")
        st.dataframe(feature_summary)

    st.sidebar.divider()
    st.sidebar.markdown("Run locally: `streamlit run dashboard.py`.")


if __name__ == "__main__":
    main()


