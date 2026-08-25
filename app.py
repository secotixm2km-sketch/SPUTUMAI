import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import datetime
import tempfile
import os
from fpdf import FPDF
import time

# =====================================================================
# 1. KONFIGURASI HALAMAN (Tampilan Pro dengan Sidebar Terbuka)
# =====================================================================
st.set_page_config(page_title="SputumAI | Clinical Workspace", page_icon="🦠", layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# 2. CSS SUPER MODERN & ANTI-DARK MODE
# =====================================================================
custom_css = """
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container {padding: 1rem 2rem 2rem 2rem; max-width: 100%;}
    .stApp { background-color: #f1f5f9; font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Fix Anti-Dark Mode */
    [data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p, 
    [data-testid="stMarkdownContainer"] span, div[role="radiogroup"] label div { color: #1e293b !important; }
    .medical-header p { color: #cbd5e1 !important; }
    .report-box p, .report-box h3, .report-box h2, .report-box div { color: inherit !important; }

    /* Banner Pro */
    .medical-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white; padding: 25px 35px; border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); margin-bottom: 25px;
        display: flex; justify-content: space-between; align-items: center; border-left: 6px solid #0ea5e9;
    }
    .medical-header h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.5px;}

    /* Card UI Modern */
    .card {
        background: #ffffff; border-radius: 16px; padding: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #f1f5f9; margin-bottom: 20px; transition: all 0.3s ease;
    }
    .card:hover { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    .card-title {
        font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 15px;
        text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;
    }

    /* Progress Bar Confidence Score */
    .progress-bg { background-color: #e2e8f0; border-radius: 999px; height: 10px; width: 100%; margin-top: 8px; overflow: hidden;}
    .progress-fill { background-color: #0ea5e9; height: 10px; border-radius: 999px; transition: width 1s ease-in-out;}
    
    /* Box Hasil Analisis */
    .report-box { padding: 20px; border-radius: 12px; margin-top: 15px; }
    .report-safe { background-color: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
    .report-warning { background-color: #fefce8; border: 1px solid #fef08a; color: #854d0e; }
    .report-danger { background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
    .report-box h2 { font-size: 2.5rem; margin: 5px 0; font-weight: 800;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =====================================================================
# 3. SIDEBAR (Workspace Pasien & Pengaturan)
# =====================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063206.png", width=60) # Ikon Medis
    st.markdown("## Data Pasien")
    id_pasien = st.text_input("No. Rekam Medis / ID", placeholder="RM-...", help="Masukkan ID Pasien untuk dicetak di PDF")
    usia_pasien = st.number_input("Usia Pasien", min_value=1, max_value=120, value=30)
    
    st.markdown("---")
    st.markdown("## Panduan Medis")
    st.info("📌 **Standar Sampel:**\n- Fokus lensa tajam.\n- Pencahayaan merata.\n- Pewarnaan Ziehl-Neelsen.\n- Resolusi min. 800x600 px.")
    
    st.markdown("---")
    st.caption("SputumAI v4.0 Pro Workspace\n© 2026")

# =====================================================================
# 4. HEADER UTAMA & LOAD MODEL
# =====================================================================
st.markdown("""
<div class="medical-header">
    <div>
        <h1>SputumAI <span style="color: #38bdf8;">Workspace</span></h1>
        <p>Sistem Pendukung Keputusan Klinis Berbasis Artificial Intelligence</p>
    </div>
    <div style="font-size: 2.5rem; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 12px;">🔬</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except:
    st.error("⚠️ Model best.pt tidak ditemukan!")
    st.stop()

# =====================================================================
# 5. AREA KERJA UTAMA (INPUT & OUTPUT)
# =====================================================================
col_input, col_output = st.columns([1, 1.8], gap="large")

# ----------------- PANEL KIRI: UPLOAD CITRA -----------------
with col_input:
    st.markdown("<div class='card'><div class='card-title'>📥 1. Input Citra Mikroskopis</div>", unsafe_allow_html=True)
    metode_input = st.radio("Metode Pengambilan:", ["📂 Unggah File", "📸 Kamera Mikroskop"], horizontal=True)
    st.write("")
    
    gambar_input = None 
    if "Unggah File" in metode_input:
        gambar_input = st.file_uploader("Upload sampel (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        gambar_input = st.camera_input("Ambil dari lensa", label_visibility="collapsed")
        
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PANEL KANAN: ANALISIS AI -----------------
with col_output:
    if gambar_input is None:
        # Empty State
        st.markdown("""
        <div class="card" style="text-align: center; padding: 80px 20px;">
            <div style="font-size: 5rem; opacity: 0.5;">🖥️</div>
            <h3 style="color: #334155;">Workspace Siap Digunakan</h3>
            <p style="color: #64748b;">Silakan lengkapi data pasien di menu samping dan unggah citra mikroskopis untuk memulai pemindaian AI.</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        image = Image.open(gambar_input).convert('RGB')
        st.markdown("<div class='card'><div class='card-title'>⚙️ 2. Proses Diagnostik</div>", unsafe_allow_html=True)
        
        run_button = st.button('🚀 JALANKAN PEMINDAIAN AI', type="primary", use_container_width=True)
        
        if not run_button:
            st.info("Citra berhasil dimuat. Klik tombol di atas untuk memulai analisis.")
            
        else:
            with st.spinner('Menjalankan algoritma deteksi BTA...'):
                # Simulasi loading tambahan sedikit agar terkesan sistem sedang berpikir keras
                time.sleep(1)
                
                # Proses YOLO
                results = model.predict(source=image, conf=0.1, imgsz=640)
                res_plotted = results[0].plot() 
                boxes = results[0].boxes
                jumlah_bakteri = len(boxes)
                
                # Kalkulasi Confidence
                if jumlah_bakteri > 0:
                    conf_list = boxes.conf.tolist()
                    avg_conf = (sum(conf_list) / len(conf_list)) * 100
                else:
                    avg_conf = 0.0
                
                st.toast('Pemindaian Selesai!', icon='✅') # Toast Notification

                # BIKIN TABBED UI UNTUK HASIL (LEBIH PROFESIONAL)
                tab1, tab2 = st.tabs(["🖼️ Analisis Visual", "📄 Laporan Klinis & PDF"])
                
                # TAB 1: GAMBAR
                with tab1:
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        st.image(image, caption="Citra Original", use_container_width=True)
                    with col_g2:
                        st.image(res_plotted, channels="BGR", caption="Deteksi Bounding Box AI", use_container_width=True)
                
                # TAB 2: LAPORAN TEXT & PROGRESS BAR
                with tab2:
                    if jumlah_bakteri == 0:
                        kat_html, box_class = "Negatif (Bersih)", "report-safe"
                        kat_pdf, intp_pdf = "Negatif", "Tidak ditemukan indikasi bakteri pada lapang pandang ini."
                    elif 1 <= jumlah_bakteri <= 9:
                        kat_html, box_class = "Positif Lemah (Scanty)", "report-warning"
                        kat_pdf, intp_pdf = "Positif (Scanty)", "Indikasi infeksi awal (1-9 BTA). Perlu observasi lebih lanjut."
                    else:
                        kat_html, box_class = "Positif Aktif (+1 / +2 / +3)", "report-danger"
                        kat_pdf, intp_pdf = "Positif Aktif", "Beban bakteri sangat tinggi. Terindikasi infeksi aktif parah."

                    # HTML Box Hasil dengan Progress Bar Confidence
                    st.markdown(f"""
                    <div class="report-box {box_class}">
                        <h4 style="margin:0; opacity:0.8;">Kategori Deteksi:</h4>
                        <h3 style="margin:5px 0 15px 0;">{kat_html}</h3>
                        <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                            <div>
                                <p style="margin:0; font-size:14px; opacity:0.8;">Total Sel Terdeteksi</p>
                                <h2>{jumlah_bakteri} <span style="font-size:20px;">BTA</span></h2>
                            </div>
                            <div style="width: 50%; text-align:right;">
                                <p style="margin:0; font-size:14px; opacity:0.8; font-weight:bold;">Tingkat Kepercayaan AI (Confidence)</p>
                                <div class="progress-bg">
                                    <div class="progress-fill" style="width: {avg_conf}%;"></div>
                                </div>
                                <p style="margin:5px 0 0 0; font-size:14px; font-weight:bold;">{avg_conf:.1f}%</p>
                            </div>
                        </div>
                        <hr style="border-color: rgba(0,0,0,0.1); margin: 15px 0;">
                        <p style="margin:0;"><strong>Interpretasi:</strong> {intp_pdf}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    
                    # ================= PEMBUATAN PDF =================
                    img_pil = Image.fromarray(res_plotted[..., ::-1]) 
                    buf = io.BytesIO()
                    img_pil.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "Laporan Klinis SputumAI", ln=True, align="C")
                    pdf.line(10, 20, 200, 20)
                    pdf.ln(5)
                    
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 6, f"ID / No. Rekam Medis : {id_pasien if id_pasien else 'Tidak Diisi'}", ln=True)
                    pdf.cell(0, 6, f"Usia Pasien          : {usia_pasien} Tahun", ln=True)
                    pdf.cell(0, 6, f"Tanggal Pemeriksaan  : {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}", ln=True)
                    pdf.ln(5)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(byte_im)
                        tmp_path = tmp.name

                    pdf.image(tmp_path, x=20, w=170)
                    pdf.ln(10)
                    
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
                    
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 8, f"Total BTA Terdeteksi : {jumlah_bakteri} Sel", ln=True)
                    pdf.cell(0, 8, f"Confidence Score     : {avg_conf:.1f}%", ln=True)
                    pdf.cell(0, 8, f"Kategori Analisis    : {kat_pdf}", ln=True)
                    pdf.cell(0, 8, f"Interpretasi Sistem  : {intp_pdf}", ln=True)
                    
                    pdf_output = pdf.output(dest='S')
                    pdf_bytes = pdf_output.encode('latin-1') if type(pdf_output) == str else bytes(pdf_output)

                    st.download_button(label="📄 Cetak Dokumen PDF", data=pdf_bytes, file_name=f"SputumAI_{id_pasien if id_pasien else 'Report'}.pdf", mime="application/pdf", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
