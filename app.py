import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import datetime
import tempfile
from fpdf import FPDF

# 1. Konfigurasi Halaman (Wajib Paling Atas)
st.set_page_config(page_title="SputumAI | Sistem Diagnostik", page_icon="🦠", layout="wide")

# =====================================================================
# 2. AREA BEBAS EDIT HTML & CSS (Silakan dimodifikasi sesuka hati!)
# =====================================================================
html_css_template = """
<style>
    /* Reset & Sembunyikan Bawaan Streamlit */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container {padding-top: 0rem; padding-bottom: 0rem;}

    /* Latar Belakang Seluruh Halaman */
    .stApp {
        background-color: #f4f7f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* HEADER / BANNER UTAMA (HTML Murni) */
    .html-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 40px 20px;
        text-align: center;
        border-radius: 0 0 25px 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .html-header h1 { margin: 0; font-size: 3rem; font-weight: 800; letter-spacing: 2px;}
    .html-header p { margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;}

    /* KARTU (CARDS) UNTUK MEMBUNGKUS KONTEN */
    .html-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .html-card-title {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        border-bottom: 3px solid #2a5298;
        padding-bottom: 10px;
        margin-bottom: 20px;
        display: inline-block;
    }

    /* KOTAK HASIL DETEKSI (Alerts) */
    .alert-box { padding: 20px; border-radius: 10px; margin-top: 15px; border-left: 8px solid; }
    .alert-safe { background-color: #e8f5e9; border-color: #2e7d32; color: #1b5e20; }
    .alert-warning { background-color: #fff3e0; border-color: #ef6c00; color: #e65100; }
    .alert-danger { background-color: #ffebee; border-color: #c62828; color: #b71c1c; }

    /* MEMAKSA TOMBOL STREAMLIT MENJADI TOMBOL HTML MODERN */
    div[data-testid="stButton"] > button {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,114,255,0.3);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,114,255,0.5);
    }

    /* MEMAKSA KOTAK UPLOAD MENJADI KEREN */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #2a5298;
        border-radius: 15px;
        background-color: #f8faff;
        padding: 10px;
    }
</style>

<!-- Elemen Header HTML Langsung -->
<div class="html-header">
    <h1>🔬 SPUTUM-AI DASHBOARD</h1>
    <p>Sistem Analisis Cerdas Mikroskopis Dahak Pasien TBC</p>
</div>
"""
st.markdown(html_css_template, unsafe_allow_html=True)
# =====================================================================


# Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# ================= KONTEN UTAMA =================
col_kiri, col_kanan = st.columns([1.2, 2])

with col_kiri:
    # Menggunakan HTML Card untuk area input
    st.markdown("<div class='html-card'><div class='html-card-title'>📥 Area Input Sampel</div>", unsafe_allow_html=True)
    
    metode_input = st.radio("Pilih Metode:", ["📂 Unggah File", "📸 Kamera Mikroskop"], horizontal=True)
    
    gambar_input = None 
    if metode_input == "📂 Unggah File":
        gambar_input = st.file_uploader("Unggah citra dahak", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        gambar_input = st.camera_input("Ambil dari Kamera", label_visibility="collapsed")
        
    st.markdown("</div>", unsafe_allow_html=True) # Tutup div html-card

with col_kanan:
    if gambar_input is not None:
        image = Image.open(gambar_input).convert('RGB')
        
        # HTML Card untuk Preview Gambar & Tombol
        st.markdown("<div class='html-card'><div class='html-card-title'>📷 Pratinjau & Analisis</div>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
        run_button = st.button('🚀 EKSEKUSI PEMINDAIAN AI', use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True) # Tutup div
        
        if run_button:
            with st.spinner('Sistem memindai morfologi bakteri...'):
                results = model.predict(source=image, conf=0.1, imgsz=640)
                res_plotted = results[0].plot() 
                jumlah_bakteri = len(results[0].boxes)
                
                # HTML Card untuk Hasil AI
                st.markdown("<div class='html-card'><div class='html-card-title'>🎯 Laporan Diagnostik AI</div>", unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns([1.5, 1])
                with col_res1:
                    st.image(res_plotted, channels="BGR", use_container_width=True)
                
                with col_res2:
                    if jumlah_bakteri == 0:
                        st.markdown(f"""
                        <div class="alert-box alert-safe">
                            <h3>✅ Negatif / Bersih</h3>
                            <h2>{jumlah_bakteri} Bakteri</h2>
                            <p>Tidak ada indikasi TBC pada area ini.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        kategori_teks = "Negatif"
                        interpretasi_teks = "Tidak ditemukan indikasi bakteri."
                    
                    elif 1 <= jumlah_bakteri <= 9:
                        st.markdown(f"""
                        <div class="alert-box alert-warning">
                            <h3>⚠️ Positif Lemah (Scanty)</h3>
                            <h2>{jumlah_bakteri} Bakteri</h2>
                            <p>Infeksi awal, butuh pemeriksaan lanjut.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        kategori_teks = "Positif Scanty"
                        interpretasi_teks = "Terindikasi positif TB tingkat awal."
                    
                    else:
                        st.markdown(f"""
                        <div class="alert-box alert-danger">
                            <h3>🚨 Positif Aktif (+1/+2/+3)</h3>
                            <h2>{jumlah_bakteri} Bakteri</h2>
                            <p>Tingkat infeksi sedang hingga parah.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        kategori_teks = "Positif Aktif"
                        interpretasi_teks = "Terindikasi infeksi aktif tingkat parah."
                
                st.markdown("</div>", unsafe_allow_html=True) # Tutup div
                
                # --- PDF GENERATOR (Tidak Diubah) ---
                img_pil = Image.fromarray(res_plotted[..., ::-1]) 
                buf = io.BytesIO()
                img_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "Laporan Analisis Skrining SputumAI", ln=True, align="C")
                pdf.line(10, 20, 200, 20)
                pdf.ln(5)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(byte_im)
                    pdf.image(tmp.name, x=20, w=170)
                pdf.ln(10)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, f"Total Sel Terdeteksi : {jumlah_bakteri} Bakteri", ln=True)
                pdf.cell(0, 8, f"Kategori / Status : {kategori_teks}", ln=True)
                pdf.cell(0, 8, f"Interpretasi : {interpretasi_teks}", ln=True)
                
                pdf_output = pdf.output(dest='S')
                pdf_bytes = pdf_output.encode('latin-1') if type(pdf_output) == str else bytes(pdf_output)

                st.download_button(label="📄 Cetak PDF Laporan Medis", data=pdf_bytes, file_name="SputumAI_Report.pdf", mime="application/pdf", use_container_width=True)
