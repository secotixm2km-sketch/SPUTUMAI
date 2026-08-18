import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Judul Website
st.title("🦠 Aplikasi Deteksi Bakteri TBC")
st.write("Unggah gambar sampel mikroskopis untuk mendeteksi *Mycobacterium tuberculosis* menggunakan AI.")

# Load Model AI (menggunakan cache agar lebih cepat)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Tombol untuk upload gambar oleh pengguna web
uploaded_file = st.file_uploader("Pilih gambar sputum...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar yang di-upload
    image = Image.open(uploaded_file)
    
    # Tampilkan gambar asli di web
    st.image(image, caption='Gambar Asli yang Diunggah', use_container_width=True)
    
    if st.button('🔍 Mulai Deteksi'):
        with st.spinner('AI sedang mendeteksi bakteri...'):
            # Jalankan prediksi model YOLO
            results = model(image)
            
            # Ambil gambar hasil deteksi yang sudah ada kotaknya (dalam bentuk numpy array)
            res_plotted = results[0].plot()
            
            # Tampilkan gambar hasil di web
            st.image(res_plotted, caption='Hasil Deteksi AI', use_container_width=True)
            st.success('Deteksi berhasil dilakukan!')