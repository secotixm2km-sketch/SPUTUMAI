import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import datetime
import tempfile
from fpdf import FPDF

# 1. Konfigurasi Halaman (Wajib Paling Atas)
st.set_page_config(page_title="SputumAI | Deteksi TBC", page_icon="🔬", layout="wide")

# 2. Custom CSS & Tema Web (Operasi Plastik UI)
custom_css = """
<style>
    /* Sembunyikan identitas bawaan Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .st-emotion-cache-1y4p8pa {padding-top: 1rem;}
    
    /* Desain Banner Utama ala Web Profesional */
    .main-banner {
        background: linear-gradient(135deg, #02aab0 0%, #00cdac 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 8px 15px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .main-banner h1 {color: white; margin-bottom: 5px; font-weight: 800; font-family: 'Arial', sans-serif; font-size: 45px;}
    .main-banner p {color: #f0f8ff; font-size: 20px; margin-top: 0; font-weight: 500;}
    
    /* Desain Kotak Hasil Diagnosa (Custom Alert HTML) */
    .result-box {padding: 15px; border-radius: 8px; margin-bottom: 20px;}
    .box-negatif {background-color: #d4edda; border-left: 6px solid #28a745; color: #155724;}
    .box-scanty {background-color: #fff3cd; border-left: 6px solid #ffc107; color: #856404;}
    .box-positif {background-color: #f8d7da; border-left: 6px solid #dc3545; color: #721c24;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Header Web Memakai HTML Banner
st.markdown("""
    <div class="main-banner">
        <h1>🔬 SPUTUM-AI SYSTEM</h1>
        <p>Platform Skrining Cerdas Sampel Dahak Mikroskopis (<i>Mycobacterium tuberculosis</i>)</p>
    </div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    # Otomatis membaca best.pt yang baru kamu masukkan
    return YOLO('best.pt')

model = load_model()

# 4. Navigasi Web
tab1, tab2 = st.tabs(["🩺 Area Kerja Diagnostik", "ℹ️ Tentang Sistem"])

# ================= TAB 1: AREA DETEKSI =================
with tab1:
    st.markdown("### 📥 Input Sampel Mikroskopis")
    st.markdown("Pilih metode input citra dahak (sputum) untuk memulai analisis.")
    
    # Pilihan Metode Input dengan tampilan Card
    metode_input = st.radio(
        "Pilih Sumber Citra:",
        ["📂 Unggah File Gambar", "📸 Kamera Mikroskop Langsung"],
        horizontal=True
    )
    
    col_input, col_instruksi = st.columns([2, 1])
    gambar_input = None 
    
    with col_input:
        if metode_input == "📂 Unggah File Gambar":
            gambar_input = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        else:
            st.info("💡 Tip: Pastikan akses kamera browser diizinkan untuk memakai mikroskop digital.")
            gambar_input = st.camera_input("Ambil citra dari mikroskop", label_visibility="collapsed")
            
    with col_instruksi:
        st.success("""
        **📌 Panduan Standar Lab:**
        - Format didukung: JPG, JPEG, PNG.
        - Pastikan pencahayaan mikroskop terang & fokus.
        - Wajib menggunakan pewarnaan standar (Ziehl-Neelsen).
        """)
        
    st.markdown("---")

    # === PROSES JIKA GAMBAR MASUK ===
    if gambar_input is not None:
        
        image = Image.open(gambar_input).convert('RGB')
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("#### 📷 Citra Original")
            st.image(image, use_container_width=True)
            run_button = st.button('🚀 Mulai Pemindaian AI', type="primary", use_container_width=True)
            
        if run_button:
            with st.spinner('🔬 AI sedang mengekstraksi fitur seluler...'):
                
                results = model.predict(source=image, conf=0.1, imgsz=640)
                res_plotted = results[0].plot() 
                jumlah_bakteri = len(results[0].boxes)
                
                with col_img2:
                    st.markdown("#### 🎯 Hasil Deteksi AI")
                    st.image(res_plotted, channels="BGR", use_container_width=True)
                    
                    st.markdown("#### 📊 Ringkasan Klinis")
                    
                    kategori_teks = ""
                    interpretasi_teks = ""
                    
                    # Logika Penentuan Status & Inject Kotak HTML Berwarna
                    if jumlah_bakteri == 0:
                        kategori_teks = "Negatif / Bersih"
                        interpretasi_teks = "Tidak ditemukan indikasi bakteri pada area citra ini. Disarankan memeriksa lapang pandang lain."
                        st.metric(label="Total Bakteri Terdeteksi", value="0 Sel", delta="Status Aman", delta_color="normal")
                        st.markdown(f"<div class='result-box box-negatif'><strong>✅ Interpretasi:</strong> {interpretasi_teks}</div>", unsafe_allow_html=True)
                        
                    elif 1 <= jumlah_bakteri <= 9:
                        kategori_teks = "Positif Lemah (Scanty)"
                        interpretasi_teks = "Ditemukan sejumlah kecil bakteri. Pasien terindikasi positif TB tingkat awal."
                        st.metric(label="Total Bakteri Terdeteksi", value=f"{jumlah_bakteri} Sel", delta="⚠️ Butuh Perhatian", delta_color="off")
                        st.markdown(f"<div class='result-box box-scanty'><strong>⚠️ Interpretasi (Scanty):</strong> {interpretasi_teks}</div>", unsafe_allow_html=True)
                        
                    elif 10 <= jumlah_bakteri <= 99:
                        kategori_teks = "Positif Aktif (+1)"
                        interpretasi_teks = "Terindikasi infeksi aktif tingkat sedang. Konsentrasi bakteri cukup tinggi."
                        st.metric(label="Total Bakteri Terdeteksi", value=f"{jumlah_bakteri} Sel", delta="🚨 Positif TB", delta_color="inverse")
                        st.markdown(f"<div class='result-box box-positif'><strong>🚨 Interpretasi (+1):</strong> {interpretasi_teks}</div>", unsafe_allow_html=True)
                        
                    else:
                        kategori_teks = "Positif Tinggi (+2 / +3)"
                        interpretasi_teks = "Terindikasi infeksi aktif parah. Beban bakteri sangat tinggi pada sampel."
                        st.metric(label="Total Bakteri Terdeteksi", value=f"{jumlah_bakteri} Sel", delta="🚨 Positif Berat", delta_color="inverse")
                        st.markdown(f"<div class='result-box box-positif'><strong>🚨 Interpretasi (+2/+3):</strong> {interpretasi_teks}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Konversi Gambar ke Biner untuk Download & PDF
                    img_pil = Image.fromarray(res_plotted[..., ::-1]) 
                    buf = io.BytesIO()
                    img_pil.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    # ================= PEMBUATAN PDF =================
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "Laporan Analisis Skrining SputumAI", ln=True, align="C")
                    
                    tanggal_sekarang = datetime.datetime.now().strftime('%d %B %Y - %H:%M')
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 10, f"Tanggal Cetak Dokumen: {tanggal_sekarang}", ln=True, align="C")
                    pdf.line(10, 30, 200, 30)
                    pdf.ln(10)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(byte_im)
                        tmp_path = tmp.name
                    pdf.image(tmp_path, x=20, w=170)
                    pdf.ln(5)
                    
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, "Ringkasan Hasil Klinis", ln=True)
                    
                    metode_pdf = "Unggah File Gambar" if "Unggah" in metode_input else "Kamera Mikroskop Langsung"
                    
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(50, 8, "Metode Input", border=0)
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 8, f": {metode_pdf}", ln=True)
                    
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(50, 8, "Total Sel Terdeteksi", border=0)
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 8, f": {jumlah_bakteri} Bakteri", ln=True)
                    
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(50, 8, "Kategori / Status", border=0)
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 8, f": {kategori_teks}", ln=True)
                    
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "Interpretasi Sistem :", ln=True)
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 7, f"{interpretasi_teks}")
                    pdf.ln(10)
                    
                    pdf.set_font("Arial", "I", 10)
                    pdf.set_text_color(100, 100, 100)
                    pdf.multi_cell(0, 5, "Peringatan (Disclaimer): Dokumen ini dihasilkan secara otomatis oleh sistem SputumAI. Ini BUKAN merupakan diagnosis medis final yang mengikat.")
                    
                    pdf_output = pdf.output(dest='S')
                    pdf_bytes = pdf_output.encode('latin-1') if type(pdf_output) == str else bytes(pdf_output)

                    # Tombol Unduh
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        st.download_button(label="🖼️ Unduh Gambar", data=byte_im, file_name="Hasil_SputumAI.jpg", mime="image/jpeg", use_container_width=True)
                    with col_btn2:
                        st.download_button(label="📄 Unduh PDF", data=pdf_bytes, file_name="Laporan_SputumAI.pdf", mime="application/pdf", type="primary", use_container_width=True)

# ================= TAB 2: TENTANG =================
with tab2:
    st.markdown("""
    <div style='background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);'>
        <h2 style='color: #02aab0;'>💻 Teknologi di Balik SputumAI</h2>
        <p style='font-size: 16px; color: #444;'>SputumAI adalah sistem cerdas berbasis Deep Learning (YOLOv8) yang telah dilatih secara khusus untuk mengenali morfologi bakteri <i>Mycobacterium tuberculosis</i> pada citra mikroskopis dengan pewarnaan Ziehl-Neelsen.</p>
        <hr>
        <p><strong>👨‍💻 Dikembangkan Oleh:</strong> [Nama Kamu / Tim]</p>
        <p><strong>🚀 Versi Model:</strong> v3 Master (50 Epochs)</p>
    </div>
    """, unsafe_allow_html=True)
