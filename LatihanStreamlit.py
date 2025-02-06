import streamlit as st
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set konfigurasi halaman
st.set_page_config(page_title="Dashboard Analisis Data", page_icon="📊", layout="wide")

# CSS Custom untuk mempercantik tampilan
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e2f;
        color: white;
        font-family: Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #25253c;
        color: white;
    }
    h1 {
        text-align: center;
        color: #00bcd4;
    }
    .stButton>button {
        background-color: #00bcd4;
        color: white;
        border-radius: 10px;
        width: 100%;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Judul aplikasi
st.markdown(
    """
    <h1>📊 Dashboard Streamlit UAS</h1>
    <hr style='border: 2px solid #00bcd4;'>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigasi
st.sidebar.image("logo.png", width=150)
st.sidebar.title("🔍 Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["📂 Unggah Dataset", "📈 Analisis Data", "📊 Visualisasi Data"],
    index=0,
    key="menu_selector",
)

# Fungsi untuk membaca dataset
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# Menu "Unggah Dataset"
if menu == "📂 Unggah Dataset":
    st.subheader("📂 Unggah Dataset")
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type="csv")
    
    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
            st.success("Dataset berhasil diunggah!")
            st.dataframe(data, height=400, width=1000)

# Menu "Analisis Data" + Data Mining
elif menu == "📈 Analisis Data":
    st.subheader("📈 Analisis Data")
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type="csv")
    
    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
            st.table(data.describe())
            
            st.subheader("🔍 Clustering dengan K-Means")
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            selected_columns = st.multiselect("Pilih dua kolom untuk clustering:", numeric_columns)
            
            if len(selected_columns) > 1:
                # Menghapus atau mengisi nilai NaN
                if st.checkbox("Hapus baris dengan nilai NaN"):
                    data = data.dropna(subset=selected_columns)
                else:
                    data[selected_columns] = data[selected_columns].fillna(data[selected_columns].mean())
                
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data[selected_columns])
                
                k = st.slider("Pilih jumlah klaster:", 2, 10, 3)
                kmeans = KMeans(n_clusters=k, random_state=42)
                
                # Pastikan tidak ada NaN setelah standarisasi
                if np.isnan(scaled_data).any():
                    st.error("Dataset masih mengandung nilai NaN setelah preprocessing. Silakan periksa kembali data Anda.")
                else:
                    clusters = kmeans.fit_predict(scaled_data)
                    data['Cluster'] = clusters
                    st.dataframe(data)
                    
                    # Visualisasi Clustering
                    if len(selected_columns) >= 2:
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=data[selected_columns[0]], y=data[selected_columns[1]], hue=data['Cluster'], palette="viridis", ax=ax)
                        ax.set_title(f"Visualisasi Clustering ({selected_columns[0]} vs {selected_columns[1]})")
                        st.pyplot(fig)

# Menu "Visualisasi Data"
elif menu == "📊 Visualisasi Data":
    st.subheader("📊 Visualisasi Data")
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type="csv")
    
    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            column = st.selectbox("Pilih Kolom Numerik", options=numeric_columns)
            visualization_type = st.selectbox("Pilih Jenis Visualisasi", ["Histogram", "Box Plot", "Bar Plot"])
            
            if visualization_type == "Histogram":
                fig, ax = plt.subplots()
                sns.histplot(data[column], bins=30, kde=True, ax=ax, color="#00bcd4")
                st.pyplot(fig)
            elif visualization_type == "Box Plot":
                fig, ax = plt.subplots()
                sns.boxplot(x=data[column], ax=ax, color="#00bcd4")
                st.pyplot(fig)
            elif visualization_type == "Bar Plot":
                fig, ax = plt.subplots()
                sns.barplot(x=data[column].value_counts().index, y=data[column].value_counts(), ax=ax, color="#00bcd4")
                st.pyplot(fig)
