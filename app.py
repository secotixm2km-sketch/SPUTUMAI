import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import datetime
import tempfile
from fpdf import FPDF

# 1. Konfigurasi Halaman 
st.set_page_config(page_title="SputumAI | Clinical Dashboard", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")

# 2. CSS MODERN CLINICAL DASHBOARD (Dengan Fix Anti-Dark Mode)
custom_css = """
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container {padding: 1.5rem 2rem; max-width: 100%;}
    
    .stApp { background-color: #f8fafc; font-family: 'Segoe UI', Roboto, sans-serif; }

    /* ======================================================== */
    /* FIX TEKS PUTIH HILANG (ANTI-DARK MODE)                   */
    /* ======================================================== */
    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stMarkdownContainer"] span, 
    div[role="radiogroup"] label div,
    .st-expanderContent p {
        color: #334155 !important;
    }
    
    /* Pengecualian agar teks di banner atas tetap putih */
    .medical-header p {
        color: #e2e8f0 !important;
    }
    /* Pengecualian teks dalam kotak hasil agar warnanya ngikut kotak */
    .report-box p, .report-box h3, .report-box h2, .report-box div {
        color: inherit !important;
    }
    /* ======================================================== */

    .medical-header {
        background: #0f172a; color: white; padding: 30px 40px; border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 25px;
        display: flex; justify-content: space-between; align-items: center; border-bottom: 5px solid #0ea5e9;
    }
    .medical-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: #ffffff;}

    .card {
        background: #ffffff; border-radius: 12px; padding: 25px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0; margin-bottom: 20px;
    }
    .card-title {
        font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-bottom: 20px;
        display: flex; align-items: center; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px;
    }

    .empty-state { text-align: center; padding: 60px 20px; background: #f8fafc; border-radius: 12px; border: 2px dashed #cbd5e1; }
    .empty-state h3 { color: #475569; margin-top: 15px;}
    .empty-state p { color: #94a3b8 !important;}

    .report-box { padding: 20px; border-radius: 8px; margin-top: 20px; border-left: 6px solid; }
    .report-safe { background-color: #f0fdf4; border-color: #22c55e; color: #166534; }
    .report-warning { background-color: #fefce8; border-color: #eab308; color: #854d0e; }
    .report-danger { background-color: #fef2f2; border-color: #ef4444; color: #991b1b; }
    .report-box h2 { font-size: 2.2rem; margin: 10px 0;}
    .conf-score { font-size: 1.1rem; font-weight: 600; margin-bottom: 10px; }
    
    .disclaimer { font-size: 0.85rem; color: #64748b !important; text-align: center; margin-top: 30px; border-top: 1px solid #e2e8f0; padding-top: 15px;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# HEADER UTAMA
st.markdown("""
<div class="medical-header">
    <div>
        <h1>SputumAI <span style="color: #38bdf8;">Diagnostics</span></h1>
        <p>Sistem Pendukung Keputusan Klinis Berbasis Artificial Intelligence</p>
    </div>
    <div style="font-size: 3rem;">🔬</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except:
    st.error("Model best.pt tidak ditemukan!")
    st.stop()

col_input, col_output = st.columns([1, 2.2], gap="large")

# BAGIAN KIRI: PANEL INPUT
with col_input:
    st.markdown("<div class='card'><div class='card-title'>📋 Metadata Pasien</div>", unsafe_allow_html=True)
    id_pasien = st.text_input("Nomor Rekam Medis / ID Anonim", placeholder="Contoh: RM-2026-08X")
    usia_pasien = st.number_input("Usia Pasien", min_value=1, max_value=120, value=30)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-title'>📥 Panel Input Citra</div>", unsafe_allow_html=True)
    metode_input = st.radio("Sumber Citra Mikroskopis:", ["📂 Unggah File (Drag & Drop)", "📸 Kamera Mikroskop"])
    
    with st.expander("💡 Lihat Panduan Kualitas Citra"):
        st.write("- Pastikan gambar fokus (tidak blur).")
        st.write("- Pencahayaan terang dan merata.")
        st.write("- Gunakan pewarnaan Ziehl-Neelsen (ZN).")
        st.write("- Resolusi disarankan: minimal 800x600 px.")
    
    gambar_input = None 
    if "Unggah File" in metode_input:
        gambar_input = st.file_uploader("Upload sampel dahak", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        gambar_input = st.camera_input("Ambil dari lensa mikroskop", label_visibility="collapsed")
        
    st.markdown("</div>", unsafe_allow_html=True)

# BAGIAN KANAN: PANEL OUTPUT
with col_output:
    if gambar_input is None:
        st.markdown("""
        <div class="card">
            <div class="card-title">🖥️ Ruang Analisis AI</div>
            <div class="empty-state">
                <div style="font-size: 4rem;">🩺</div>
                <h3>Menunggu Input Citra Medis</h3>
                <p>Silakan lengkapi metadata dan unggah sampel dahak di panel sebelah kiri (mendukung fitur Drag & Drop).</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        image = Image.open(gambar_input).convert('RGB')
        st.markdown("<div class='card'><div class='card-title'>🖥️ Hasil Analisis AI</div>", unsafe_allow_html=True)
        
        run_button = st.button('🚀 EKSEKUSI PEMINDAIAN', type="primary", use_container_width=True)
        
        if not run_button:
            st.image(image, caption="Pratinjau Citra Original", use_container_width=True)
            
        else:
            with st.spinner('AI sedang memetakan morfologi bakteri...'):
                results = model.predict(source=image, conf=0.1, imgsz=640)
                res_plotted = results[0].plot() 
                
                boxes = results[0].boxes
                jumlah_bakteri = len(boxes)
                
                if jumlah_bakteri > 0:
                    conf_list = boxes.conf.tolist()
                    avg_conf = (sum(conf_list) / len(conf_list)) * 100
                    conf_text = f"{avg_conf:.1f}%"
                else:
                    conf_text = "N/A"
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.image(image, caption="Citra Original", use_container_width=True)
                with col_g2:
                    st.image(res_plotted, channels="BGR", caption="Deteksi AI (Bounding Box)", use_container_width=True)
                
                if jumlah_bakteri == 0:
                    st.markdown(f"""
                    <div class="report-box report-safe">
                        <h3>✅ Hasil: Negatif (Bersih)</h3>
                        <h2>{jumlah_bakteri} Sel BTA</h2>
                        <div class="conf-score">Confidence Score: {conf_text}</div>
                        <p><strong>Interpretasi:</strong> Tidak ditemukan indikasi bakteri pada lapang pandang ini.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    kat_pdf, intp_pdf = "Negatif", "Tidak ditemukan indikasi bakteri."
                    
                elif 1 <= jumlah_bakteri <= 9:
                    st.markdown(f"""
                    <div class="report-box report-warning">
                        <h3>⚠️ Hasil: Positif Lemah (Scanty)</h3>
                        <h2>{jumlah_bakteri} Sel BTA</h2>
                        <div class="conf-score">Confidence Score: {conf_text}</div>
                        <p><strong>Interpretasi:</strong> Indikasi infeksi awal (1-9 BTA).</p>
                    </div>
                    """, unsafe_allow_html=True)
                    kat_pdf, intp_pdf = "Positif (Scanty)", "Indikasi infeksi awal."
                    
                else:
                    st.markdown(f"""
                    <div class="report-box report-danger">
                        <h3>🚨 Hasil: Positif Aktif (+1 / +2 / +3)</h3>
                        <h2>{jumlah_bakteri} Sel BTA</h2>
                        <div class="conf-score">Confidence Score: {conf_text}</div>
                        <p><strong>Interpretasi:</strong> Beban bakteri sangat tinggi. Terindikasi infeksi aktif.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    kat_pdf, intp_pdf = "Positif Aktif", "Terindikasi infeksi tingkat parah."
                
                st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
                
                # PEMBUATAN PDF
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
                    pdf.image(tmp.name, x=20, w=170)
                pdf.ln(10)
                
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, f"Total BTA Terdeteksi : {jumlah_bakteri} Sel", ln=True)
                pdf.cell(0, 8, f"Confidence Score     : {conf_text}", ln=True)
                pdf.cell(0, 8, f"Kategori Analisis    : {kat_pdf}", ln=True)
                pdf.cell(0, 8, f"Interpretasi Sistem  : {intp_pdf}", ln=True)
                
                pdf_output = pdf.output(dest='S')
                pdf_bytes = pdf_output.encode('latin-1') if type(pdf_output) == str else bytes(pdf_output)

                st.download_button(label="📄 Cetak PDF Laporan Medis", data=pdf_bytes, file_name="SputumAI_Report.pdf", mime="application/pdf", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <strong>DISCLAIMER MEDIS:</strong> Aplikasi SputumAI adalah alat bantu deteksi berbasis Artificial Intelligence dan tidak menggantikan keputusan medis profesional. 
    Hasil pemindaian harus divalidasi oleh tenaga medis yang berwenang sebelum digunakan untuk diagnosis akhir atau rencana pengobatan.
</div>
""", unsafe_allow_html=True)
