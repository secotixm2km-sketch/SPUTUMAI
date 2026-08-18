import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Konfigurasi Halaman (Harus di baris paling atas)
st.set_page_config(page_title="TBC AI Detector", page_icon="🦠", layout="wide")

# 2. Membuat Sidebar untuk Informasi Tambahan
with st.sidebar:
    st.title("🔬 Tentang Aplikasi")
    st.info("Aplikasi ini menggunakan teknologi *Machine Learning* (YOLOv8) untuk mendeteksi keberadaan bakteri *Mycobacterium tuberculosis* pada sampel dahak (sputum) mikroskopis.")
    st.warning("⚠️ **Perhatian:** Hasil analisis AI ini hanya untuk tujuan penelitian/skrining awal dan tidak menggantikan diagnosis resmi dari tenaga medis.")
    st.markdown("---")
    st.markdown("Dibuat oleh: **[Nama Kamu/Tim]**")

# Judul Utama
st.title("🦠 Sistem Deteksi Bakteri TBC Cerdas")
st.markdown("Silakan unggah gambar sampel mikroskopis untuk memulai analisis otomatis.")
st.markdown("---") # Garis pembatas

# Load Model AI
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Uploader Gambar
uploaded_file = st.file_uploader("Pilih file gambar sampel (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # 3. Membuat Layout 2 Kolom (Kiri & Kanan)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.image(image, use_container_width=True)
        
        # Tombol deteksi ditaruh di bawah gambar asli
        run_button = st.button('🔍 Analisis Gambar Sekarang', type="primary", use_container_width=True)
        
    if run_button:
        with st.spinner('Memproses gambar dengan AI...'):
            # Prediksi
            results = model(image)
            res_plotted = results[0].plot()
            
            # 4. Menghitung jumlah objek (bakteri) yang dideteksi
            jumlah_bakteri = len(results[0].boxes)
            
            with col2:
                st.subheader("✨ Hasil Analisis AI")
                st.image(res_plotted, use_container_width=True)
                
                # Menampilkan metrik hasil yang estetik
                if jumlah_bakteri > 0:
                    st.error(f"**Peringatan:** Ditemukan indikasi {jumlah_bakteri} bakteri *Mycobacterium tuberculosis*!")
                else:
                    st.success("**Aman:** Tidak terdeteksi adanya bakteri pada sampel ini.")
