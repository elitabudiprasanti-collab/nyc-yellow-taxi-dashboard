# ============================================
# NYC YELLOW TAXI DASHBOARD
# Advanced Data Analysis - S2 Statistics
# ============================================

# =========================
# IMPORT LIBRARY
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gdown
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

from scipy.stats import pearsonr
import hdbscan

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NYC Yellow Taxi Dashboard",
    layout="wide"
)

st.title("🚖 NYC Yellow Taxi Trip Dashboard")
st.markdown("### Advanced Data Analysis & Spatial Clustering")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    file_id = "158TsvYRDH6qzWukzAj-lRtsvXQsw61Xp"

    output = "yellow_tripdata_2010-09.parquet"

    if not os.path.exists(output):

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, output, quiet=False)

    df = pd.read_parquet(output)

    return df

# INI WAJIB ADA
df = load_data()

# =========================
# PREPROCESSING
# =========================
@st.cache_data
def preprocess_data(df):

    # Rename columns
    df = df.rename(columns={
        'pickup_longitude': 'pickup_lon',
        'pickup_latitude': 'pickup_lat',
        'dropoff_longitude': 'dropoff_lon',
        'dropoff_latitude': 'dropoff_lat',
        'pickup_datetime': 'pickup_time',
        'dropoff_datetime': 'dropoff_time'
    })

    # Datetime conversion
    df['pickup_time'] = pd.to_datetime(df['pickup_time'])
    df['dropoff_time'] = pd.to_datetime(df['dropoff_time'])

    # Trip duration
    df['trip_duration'] = (
        df['dropoff_time'] - df['pickup_time']
    ).dt.total_seconds() / 60

    # Additional features
    df['hour'] = df['pickup_time'].dt.hour
    df['day'] = df['pickup_time'].dt.day_name()

    # NYC geographic bounds
    nyc_bounds = {
        'lon_min': -74.26,
        'lon_max': -73.70,
        'lat_min': 40.50,
        'lat_max': 40.92
    }

    # Geographic filtering
    df_clean = df[
        (df['pickup_lon'].between(
            nyc_bounds['lon_min'],
            nyc_bounds['lon_max']
        )) &

        (df['pickup_lat'].between(
            nyc_bounds['lat_min'],
            nyc_bounds['lat_max']
        )) &

        (df['dropoff_lon'].between(
            nyc_bounds['lon_min'],
            nyc_bounds['lon_max']
        )) &

        (df['dropoff_lat'].between(
            nyc_bounds['lat_min'],
            nyc_bounds['lat_max']
        ))
    ].copy()

    # Remove outliers
    df_clean = df_clean[
        (df_clean['trip_distance'] > 0) &
        (df_clean['trip_distance'] < 50) &

        (df_clean['fare_amount'] > 0) &
        (df_clean['fare_amount'] < 500) &

        (df_clean['passenger_count'] > 0) &
        (df_clean['passenger_count'] <= 6) &

        (df_clean['trip_duration'] > 0) &
        (df_clean['trip_duration'] < 300)
    ]

    # Missing values
    df_clean = df_clean.dropna()

    return df_clean

df_clean = preprocess_data(df)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Dashboard Settings")

sample_size = st.sidebar.slider(
    "Sample Size",
    min_value=5000,
    max_value=50000,
    value=20000,
    step=5000
)

method = st.sidebar.selectbox(
    "Choose Clustering Method",
    ["HDBSCAN", "KMeans", "DBSCAN", "KMedoids"]
)

# =========================
# SAMPLE DATA
# =========================
df_sample = df_clean.sample(
    n=sample_size,
    random_state=42
)

coords = df_sample[['pickup_lon', 'pickup_lat']].values

# Scaling
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# =========================
# OVERVIEW
# =========================
st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Trips",
    f"{len(df_clean):,}"
)

col2.metric(
    "Average Distance",
    f"{df_clean['trip_distance'].mean():.2f} miles"
)

col3.metric(
    "Average Fare",
    f"${df_clean['fare_amount'].mean():.2f}"
)

col4.metric(
    "Average Duration",
    f"{df_clean['trip_duration'].mean():.2f} min"
)

# =========================
# EDA SECTION
# =========================
st.header("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df_sample,
        x='trip_distance',
        nbins=50,
        title='Trip Distance Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.histogram(
        df_sample,
        x='fare_amount',
        nbins=50,
        title='Fare Amount Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# CORRELATION
# =========================
st.subheader("🔗 Correlation Analysis")

corr = df_sample[
    ['trip_distance',
     'fare_amount',
     'trip_duration',
     'passenger_count']
].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Correlation Heatmap"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# DENSITY MAP
# =========================
st.header("🗺️ Pickup Density Map")

fig = px.density_mapbox(
    df_sample,
    lat='pickup_lat',
    lon='pickup_lon',
    radius=8,
    zoom=10,
    height=600,
    mapbox_style="carto-positron"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# CLUSTERING
# =========================
st.header("🧠 Spatial Clustering Analysis")

start = time.time()

# =========================
# HDBSCAN
# =========================
if method == "HDBSCAN":

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    labels = clusterer.fit_predict(coords_scaled)

# =========================
# KMEANS
# =========================
elif method == "KMeans":

    clusterer = KMeans(
        n_clusters=8,
        random_state=42
    )

    labels = clusterer.fit_predict(coords_scaled)

# =========================
# DBSCAN
# =========================
elif method == "DBSCAN":

    clusterer = DBSCAN(
        eps=0.15,
        min_samples=10
    )

    labels = clusterer.fit_predict(coords_scaled)

# =========================
# KMEDOIDS
# =========================
elif method == "KMedoids":

    clusterer = KMedoids(
        n_clusters=8,
        random_state=42,
        method='pam'
    )

    labels = clusterer.fit_predict(coords_scaled)

# =========================
# RESULT
# =========================
exec_time = time.time() - start

df_sample['cluster'] = labels

# Noise handling
if -1 in labels:

    noise_pct = (
        (labels == -1).sum() / len(labels)
    ) * 100

else:
    noise_pct = 0

# Valid cluster count
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Evaluation metrics
valid_mask = labels != -1

if len(set(labels[valid_mask])) > 1:

    silhouette = silhouette_score(
        coords_scaled[valid_mask],
        labels[valid_mask]
    )

    dbi = davies_bouldin_score(
        coords_scaled[valid_mask],
        labels[valid_mask]
    )

else:

    silhouette = np.nan
    dbi = np.nan

# =========================
# METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Clusters",
    n_clusters
)

col2.metric(
    "Noise %",
    f"{noise_pct:.2f}%"
)

col3.metric(
    "Silhouette",
    f"{silhouette:.3f}"
)

col4.metric(
    "Davies-Bouldin",
    f"{dbi:.3f}"
)

st.success(
    f"⏱️ Clustering completed in {exec_time:.2f} seconds"
)

# =========================
# CLUSTER MAP
# =========================
fig = px.scatter_mapbox(
    df_sample,
    lat='pickup_lat',
    lon='pickup_lon',
    color=df_sample['cluster'].astype(str),
    zoom=10,
    height=700,
    hover_data=[
        'trip_distance',
        'fare_amount',
        'trip_duration'
    ],
    mapbox_style='carto-positron'
)

fig.update_traces(
    marker=dict(size=4, opacity=0.6)
)

fig.update_layout(
    title=f"{method} Clustering Result"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# PEAK HOUR ANALYSIS
# =========================
st.header("⏰ Peak Hour Analysis")

hourly = df_clean.groupby('hour').size().reset_index(name='total_trip')

fig = px.line(
    hourly,
    x='hour',
    y='total_trip',
    markers=True,
    title='Taxi Trip by Hour'
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# STATISTICAL ANALYSIS
# =========================
st.header("📚 Statistical Insight")

r1, p1 = pearsonr(
    df_sample['trip_distance'],
    df_sample['fare_amount']
)

r2, p2 = pearsonr(
    df_sample['trip_distance'],
    df_sample['trip_duration']
)

st.write(
    f"""
### Correlation Results

- Distance vs Fare:
  r = {r1:.3f}, p-value = {p1:.5f}

- Distance vs Duration:
  r = {r2:.3f}, p-value = {p2:.5f}
"""
)

# =========================
# INTERPRETATION
# =========================
st.header("📝 Interpretation")

st.markdown(f"""

### Clustering Interpretation

- Method used: **{method}**
- Number of clusters detected: **{n_clusters}**
- Noise percentage: **{noise_pct:.2f}%**
- Silhouette Score: **{silhouette:.3f}**
- Davies-Bouldin Index: **{dbi:.3f}**

### Conclusion

The clustering analysis shows spatial mobility patterns
of NYC taxi trips.

HDBSCAN and DBSCAN are able to detect noise and irregular
spatial structures better than centroid-based methods.

The analysis also indicates that trip distance has strong
positive correlations with fare amount and trip duration.

This dashboard provides an interactive framework for
advanced spatial transportation analytics.
""")