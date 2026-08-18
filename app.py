import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# 1. Konfigurasi Halaman
st.set_page_config(page_title="SputumAI | Deteksi TBC", layout="wide")

# 2. Custom CSS untuk membersihkan antarmuka
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .st-emotion-cache-1y4p8pa {padding-top: 2rem;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 3. Header Web
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>SputumAI Analytics System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 18px;'>Platform Skrining Cerdas Sampel Dahak Mikroskopis (<i>Mycobacterium tuberculosis</i>)</p>", unsafe_allow_html=True)
st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 4. Navigasi Web
tab1, tab2 = st.tabs(["Area Kerja Diagnostik", "Tentang Sistem"])

# ================= TAB 1: AREA DETEKSI =================
with tab1:
    st.markdown("#### Area Pengunggahan Citra")
    st.markdown("Silakan unggah citra mikroskopis dahak (sputum) untuk memulai analisis. Pastikan citra memiliki resolusi yang baik dan fokus agar sistem dapat mengekstraksi fitur seluler secara optimal.")
    
    # Membagi layout menjadi 2 kolom (Kiri lebih lebar dari Kanan)
    col_upload, col_instruksi = st.columns([2, 1])
    
    with col_upload:
        # Kotak upload utama
        uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
    with col_instruksi:
        # Kotak informasi di sebelah kanan uploader
        st.info("""
        **Panduan Standar:**
        - Format didukung: JPG, JPEG, PNG
        - Pastikan pencahayaan mikroskop memadai.
        - Disarankan menggunakan sampel dengan pewarnaan standar (misal: Ziehl-Neelsen).
        """)
        
    st.markdown("---") # Garis pembatas pemisah area kerja

    # === LOGIKA JIKA GAMBAR SUDAH DIUNGGAH ===
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Citra Mikroskopis (Original)")
            st.image(image, use_container_width=True)
            
            run_button = st.button('Mulai Pemindaian', type="primary", use_container_width=True)
            
if run_button:
            with st.spinner('Sistem sedang mengekstraksi fitur seluler...'):
                results = model(image)
                res_plotted = results[0].plot() 
                
                jumlah_bakteri = len(results[0].boxes)
                
                with col2:
                    st.success("✅ Pemindaian Selesai")
                    st.image(res_plotted, channels="BGR", use_container_width=True)
                    
                    st.write("#### 📊 Ringkasan & Interpretasi Medis")
                    
                    # Logika Klasifikasi Tingkat Keparahan (Adaptasi Skala BTA)
                    if jumlah_bakteri == 0:
                        st.metric(label="Total Sel Terdeteksi", value="0 Bakteri", delta="Negatif / Bersih", delta_color="normal")
                        st.info("**Interpretasi:** Tidak ditemukan indikasi bakteri *Mycobacterium tuberculosis* pada area citra ini. Disarankan memeriksa lapang pandang sampel lainnya untuk memastikan hasil diagnosis.")
                    
                    elif 1 <= jumlah_bakteri <= 9:
                        st.metric(label="Total Sel Terdeteksi", value=f"{jumlah_bakteri} Bakteri", delta="Positif Lemah (Scanty)", delta_color="inverse")
                        st.warning("**Interpretasi (Scanty):** Ditemukan sejumlah kecil bakteri. Pasien terindikasi positif TB tingkat awal. Sangat disarankan untuk melakukan pengujian ulang sampel atau konfirmasi dengan Tes Cepat Molekuler (TCM).")
                    
                    elif 10 <= jumlah_bakteri <= 99:
                        st.metric(label="Total Sel Terdeteksi", value=f"{jumlah_bakteri} Bakteri", delta="Positif Aktif (+1)", delta_color="inverse")
                        st.error("**Interpretasi (+1):** Terindikasi infeksi aktif tingkat sedang. Konsentrasi bakteri cukup tinggi pada dahak. Segera rujuk pasien ke fasilitas layanan kesehatan untuk memulai pengobatan OAT (Obat Anti Tuberkulosis).")
                    
                    else:
                        st.metric(label="Total Sel Terdeteksi", value=f"{jumlah_bakteri} Bakteri", delta="Positif Tinggi (+2 / +3)", delta_color="inverse")
                        st.error("**Interpretasi (+2 / +3):** Terindikasi infeksi aktif tingkat parah. Beban bakteri sangat tinggi dan pasien berisiko tinggi menularkan penyakit. Ambil tindakan isolasi dan intervensi medis darurat segera.")
                    
                    st.markdown("---")
                    
                    # Fitur Download Gambar
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
    st.write("### Teknologi di Balik SputumAI")
    st.write("""
    SputumAI adalah prototipe sistem cerdas yang dirancang untuk membantu tenaga medis mempercepat skrining penyakit Tuberculosis (TBC). 
    Sistem ini ditenagai oleh arsitektur *Deep Learning* mutakhir yaitu YOLOv8, yang dilatih khusus untuk mengenali bentuk dan pola bakteri *Mycobacterium tuberculosis* dari citra mikroskop cahaya.
    """)
    st.info("Dikembangkan Oleh: **[Isi dengan Nama Kamu atau Tim]**")
    
    st.warning("""
    **Peringatan Medis (Disclaimer):**
    Platform ini masih dalam tahap Alpha dan murni digunakan untuk tujuan riset akademis. Tingkat akurasi mesin dapat bervariasi bergantung pada kualitas pewarnaan sampel dan pencahayaan mikroskop. Hasil pemindaian AI tidak sah digunakan sebagai diagnosis final tanpa validasi dari dokter spesialis patologi atau tenaga kesehatan berwenang.
    """)
