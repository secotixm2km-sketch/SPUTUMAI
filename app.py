import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# 1. Konfigurasi Halaman (Bikin Tab Browser lebih pro)
st.set_page_config(page_title="SputumAI | Deteksi TBC", page_icon="🔬", layout="wide")

# 2. Custom CSS untuk menghapus elemen bawaan Streamlit (Biar gak kelihatan pakai template)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .st-emotion-cache-1y4p8pa {padding-top: 2rem;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 3. Header Web bergaya Profesional
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🔬 SputumAI Analytics System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 18px;'>Platform Skrining Cerdas Sampel Dahak Mikroskopis (<i>Mycobacterium tuberculosis</i>)</p>", unsafe_allow_html=True)
st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 4. Membuat Menu Navigasi Web menggunakan Tab
tab1, tab2 = st.tabs(["🔍 Area Kerja Diagnostik", "📖 Tentang SputumAI"])

# ================= TAB 1: AREA DETEKSI =================
with tab1:
    st.write("#### 📤 Unggah Sampel")
    
    # Uploader yang lebih lebar dan bersih
    uploaded_file = st.file_uploader("Pilih file gambar dengan resolusi baik", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Grid yang rapi
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("📌 Citra Mikroskopis (Original)")
            # Membatasi tampilan gambar agar tidak kebesaran/pecah
            st.image(image, use_container_width=True)
            
            run_button = st.button('🚀 Mulai Pemindaian AI', type="primary", use_container_width=True)
            
        if run_button:
            with st.spinner('Sistem AI sedang mengekstraksi fitur seluler...'):
                # Prediksi
                results = model(image)
                res_plotted = results[0].plot() # Ini masih format BGR dari OpenCV
                
                # Menghitung bakteri
                jumlah_bakteri = len(results[0].boxes)
                
                with col2:
                    st.success("✅ Pemindaian Selesai")
                    st.image(res_plotted, channels="BGR", use_container_width=True)
                    
                    # Tampilan data Dashboard Metrik
                    st.write("#### 📊 Ringkasan Analisis")
                    if jumlah_bakteri > 0:
                        st.metric(label="Total Sel Terdeteksi", value=f"{jumlah_bakteri} Bakteri", delta="Terindikasi TB Positif", delta_color="inverse")
                    else:
                        st.metric(label="Total Sel Terdeteksi", value="0 Bakteri", delta="Negatif / Bersih", delta_color="normal")
                    
                    # Fitur Ekspor Gambar (Convert ke RGB agar warna tidak terbalik saat didownload)
                    img_pil = Image.fromarray(res_plotted[..., ::-1]) 
                    buf = io.BytesIO()
                    img_pil.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Unduh Citra Hasil Analisis",
                        data=byte_im,
                        file_name="SputumAI_Result.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )

# ================= TAB 2: TENTANG =================
with tab2:
    st.write("### 🧠 Teknologi di Balik SputumAI")
    st.write("""
    SputumAI adalah prototipe sistem cerdas yang dirancang untuk membantu tenaga medis mempercepat skrining penyakit Tuberculosis (TBC). 
    Sistem ini ditenagai oleh arsitektur *Deep Learning* mutakhir yaitu **YOLOv8**, yang dilatih khusus untuk mengenali bentuk dan pola bakteri *Mycobacterium tuberculosis* dari citra mikroskop cahaya.
    """)
    st.info("👨‍💻 Didevelop Oleh: **[Isi dengan Nama Kamu atau Tim]**")
    
    st.warning("""
    **🛑 Peringatan Medis (Disclaimer):**
    Platform ini masih dalam tahap *Alpha* dan murni digunakan untuk tujuan riset akademis. Tingkat akurasi mesin dapat bervariasi bergantung pada kualitas pewarnaan sampel dan pencahayaan mikroskop. Hasil pemindaian AI **tidak sah** digunakan sebagai diagnosis final tanpa validasi dari dokter spesialis patologi atau tenaga kesehatan berwenang.
    """)
