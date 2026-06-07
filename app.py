# app.py - Laptop Recommendation System
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Laptop Recommendation System",
    page_icon="💻",
    layout="wide"
)

# Title
st.title("💻 Laptop Recommendation System")
st.markdown("### Temukan laptop terbaik sesuai budget dan kebutuhan Anda!")
st.markdown("---")

# Load model dan data
@st.cache_resource
def load_models():
    knn_model = joblib.load('laptop_recommender_model.pkl')
    scaler = joblib.load('laptop_scaler.pkl')
    label_encoders = joblib.load('laptop_label_encoders.pkl')
    return knn_model, scaler, label_encoders

@st.cache_data
def load_data():
    df = pd.read_pickle('laptop_data_clean.pkl')
    return df

# Load semua file
df_clean = load_data()
knn_model, scaler, label_encoders = load_models()

# Sidebar filters
st.sidebar.header("🔍 Filter Pencarian")
st.sidebar.markdown("---")

# Budget filter
min_price = int(df_clean['Price'].min())
max_price = int(df_clean['Price'].max())
budget = st.sidebar.slider(
    "💰 Budget Maksimal (₹)",
    min_value=min_price,
    max_value=max_price,
    value=min(50000, max_price),
    step=5000,
    format="₹%d"
)

# RAM filter
ram_min = st.sidebar.selectbox(
    "💾 RAM Minimal (GB)",
    options=[None, 4, 8, 16, 32, 64],
    format_func=lambda x: "Semua" if x is None else f"{x} GB"
)

# CPU filter
cpu_brand = st.sidebar.selectbox(
    "⚙️ Merek CPU",
    options=[None, 'Intel', 'AMD', 'Other'],
    format_func=lambda x: "Semua" if x is None else x
)

# GPU filter
gpu_brand = st.sidebar.selectbox(
    "🎮 Merek GPU",
    options=[None, 'NVIDIA', 'AMD', 'Intel'],
    format_func=lambda x: "Semua" if x is None else x
)

# Screen size filter
min_inches = float(df_clean['Inches'].min())
max_inches = float(df_clean['Inches'].max())
screen_size = st.sidebar.slider(
    "📺 Ukuran Layar Minimal (inci)",
    min_value=min_inches,
    max_value=max_inches,
    value=min_inches,
    step=0.1
)

# OS filter
os_type = st.sidebar.selectbox(
    "💻 Sistem Operasi",
    options=[None, 'Windows', 'Mac', 'Other'],
    format_func=lambda x: "Semua" if x is None else x
)

# Number of recommendations
n_recs = st.sidebar.slider("📊 Jumlah Rekomendasi", 3, 10, 5)

# Search button
search_button = st.sidebar.button("🔍 Cari Laptop", type="primary", use_container_width=True)

# Recommendation function
def recommend_laptops(price_max, ram_min=None, cpu_brand=None, gpu_brand=None,
                      screen_size_min=None, os_type=None, n_recommendations=5):
    
    filtered_df = df_clean[df_clean['Price'] <= price_max].copy()
    
    if ram_min:
        filtered_df = filtered_df[filtered_df['RAM_GB'] >= ram_min]
    if cpu_brand:
        filtered_df = filtered_df[filtered_df['CPU_Brand'] == cpu_brand]
    if gpu_brand:
        filtered_df = filtered_df[filtered_df['GPU_Brand'] == gpu_brand]
    if screen_size_min:
        filtered_df = filtered_df[filtered_df['Inches'] >= screen_size_min]
    if os_type:
        filtered_df = filtered_df[filtered_df['OS_Type'] == os_type]
    
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    recommendations = filtered_df.sort_values('Price').head(n_recommendations)
    return recommendations.reset_index(drop=True)

# Display results
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📋 Ringkasan Filter")
    st.markdown(f"**💰 Budget:** ₹{budget:,}")
    st.markdown(f"**💾 RAM:** {f'Minimal {ram_min} GB' if ram_min else 'Semua'}")
    st.markdown(f"**⚙️ CPU:** {cpu_brand if cpu_brand else 'Semua'}")
    st.markdown(f"**🎮 GPU:** {gpu_brand if gpu_brand else 'Semua'}")
    st.markdown(f"**📺 Layar:** Minimal {screen_size}\"")
    st.markdown(f"**💻 OS:** {os_type if os_type else 'Semua'}")

with col2:
    if search_button:
        with st.spinner("Sedang mencari laptop terbaik..."):
            results = recommend_laptops(
                price_max=budget,
                ram_min=ram_min,
                cpu_brand=cpu_brand,
                gpu_brand=gpu_brand,
                screen_size_min=screen_size,
                os_type=os_type,
                n_recommendations=n_recs
            )
            
            if len(results) > 0:
                st.markdown(f"### 🎯 Hasil Rekomendasi")
                st.markdown(f"Ditemukan **{len(results)}** laptop yang sesuai!")
                
                for idx, row in results.iterrows():
                    with st.expander(f"💻 Rekomendasi #{idx + 1} - ₹{row['Price']:,.0f}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**💰 Harga:** ₹{row['Price']:,.0f}")
                            st.markdown(f"**💾 RAM:** {row['RAM_GB']:.0f} GB")
                            st.markdown(f"**⚙️ CPU:** {row['CPU_Brand']}")
                            st.markdown(f"**🎮 GPU:** {row['GPU_Brand']}")
                        with col_b:
                            st.markdown(f"**📺 Layar:** {row['Inches']:.1f}\"")
                            st.markdown(f"**💻 OS:** {row['OS_Type']}")
                            st.markdown(f"**🖥️ Resolusi:** {row['Screen_Resolution_Type']}")
            else:
                st.error("❌ Tidak ada laptop yang sesuai!")
                st.markdown("💡 **Saran:** Coba tingkatkan budget atau kurangi filter")
    else:
        st.info("👈 Atur filter dan klik 'Cari Laptop' untuk melihat rekomendasi")

# Footer statistics
st.markdown("---")
st.markdown("### 📊 Statistik Dataset")

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
with col_stat1:
    st.metric("Total Laptop", f"{len(df_clean):,}")
with col_stat2:
    st.metric("Rata-rata Harga", f"₹{df_clean['Price'].mean():,.0f}")
with col_stat3:
    st.metric("Harga Termurah", f"₹{df_clean['Price'].min():,.0f}")
with col_stat4:
    st.metric("Harga Termahal", f"₹{df_clean['Price'].max():,.0f}")

# Price distribution chart
st.markdown("### 📈 Distribusi Harga Laptop")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_clean['Price'], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
ax.set_xlabel('Harga (₹)')
ax.set_ylabel('Jumlah Laptop')
ax.set_title('Distribusi Harga Laptop')
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & Scikit-learn")
