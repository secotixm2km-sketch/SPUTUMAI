# =============================================================================
# SputumAI Workspace
# Aplikasi Diagnostik Medis Berbasis AI untuk Deteksi Bakteri Mycobacterium
# Tuberculosis (BTA) pada Citra Dahak Mikroskopis menggunakan YOLOv8
# =============================================================================
# Stack: Streamlit, Ultralytics YOLOv8, PIL, FPDF
# Author: Senior Full-Stack HealthTech Developer & UI/UX Expert
# =============================================================================

import os
import io
import tempfile
from datetime import datetime

import streamlit as st
from PIL import Image
from fpdf import FPDF

# Ultralytics diimpor secara "lazy" agar aplikasi tetap bisa dibuka
# (misalnya untuk demo UI) walaupun dependency belum lengkap di environment.
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# =============================================================================
# 1. KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="SputumAI Workspace | Clinical Diagnostic Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best.pt"


# =============================================================================
# 2. CUSTOM CSS — CLINICAL DASHBOARD THEME (LIGHT MODE ENFORCED)
# =============================================================================
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* -----------------------------------------------------------------
           0. FORCE LIGHT MODE — mencegah teks hilang saat browser Dark Mode
        ----------------------------------------------------------------- */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            color-scheme: light only;
            background-color: #f1f5f9 !important;
            color: #1e293b !important;
        }

        [data-testid="stAppViewContainer"] * {
            color: #1e293b;
        }

        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0) !important;
        }

        /* Sembunyikan elemen bawaan Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}

        /* -----------------------------------------------------------------
           1. TYPOGRAPHY & GLOBAL
        ----------------------------------------------------------------- */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1300px;
        }

        /* -----------------------------------------------------------------
           2. HEADER BANNER — Navy Blue Gradient
        ----------------------------------------------------------------- */
        .clinical-header {
            background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #0ea5e9 130%);
            padding: 28px 36px;
            border-radius: 18px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
            position: relative;
            overflow: hidden;
        }
        .clinical-header::after {
            content: "";
            position: absolute;
            top: -60px; right: -60px;
            width: 220px; height: 220px;
            background: rgba(14, 165, 233, 0.25);
            border-radius: 50%;
        }
        .clinical-header h1 {
            color: #ffffff !important;
            font-size: 30px;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .clinical-header p {
            color: #dbeafe !important;
            font-size: 14.5px;
            margin: 6px 0 0 0;
            font-weight: 400;
            max-width: 720px;
        }
        .clinical-header .badge-row {
            margin-top: 14px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .header-badge {
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.25);
            color: #e0f2fe !important;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            backdrop-filter: blur(4px);
        }

        /* -----------------------------------------------------------------
           3. CARD COMPONENT — SaaS Dashboard Style
        ----------------------------------------------------------------- */
        .card {
            background: #ffffff;
            border-radius: 16px;
            padding: 22px 24px;
            box-shadow: 0 4px 18px rgba(30, 41, 59, 0.07);
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }
        .card-title {
            font-size: 15.5px;
            font-weight: 700;
            color: #0f172a !important;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card-subtitle {
            font-size: 12.5px;
            color: #64748b !important;
            margin-bottom: 16px;
        }

        /* -----------------------------------------------------------------
           4. SIDEBAR
        ----------------------------------------------------------------- */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0;
        }
        [data-testid="stSidebar"] * {
            color: #1e293b !important;
        }
        [data-testid="stSidebar"] .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 0 18px 0;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 18px;
        }
        [data-testid="stSidebar"] .sidebar-brand-icon {
            background: linear-gradient(135deg, #0f172a, #0ea5e9);
            width: 38px; height: 38px;
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 19px;
        }
        [data-testid="stSidebar"] .sidebar-brand-text b {
            font-size: 15px;
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] .sidebar-brand-text span {
            font-size: 11px;
            color: #64748b !important;
        }
        .sidebar-section-label {
            font-size: 11.5px;
            font-weight: 700;
            color: #0ea5e9 !important;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin: 18px 0 8px 0;
        }
        .guidance-box {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 12.5px;
            color: #0c4a6e !important;
            line-height: 1.6;
        }
        .guidance-box ul {
            margin: 6px 0 0 0;
            padding-left: 18px;
        }
        .guidance-box b { color: #0c4a6e !important; }

        /* -----------------------------------------------------------------
           5. INPUT WIDGETS
        ----------------------------------------------------------------- */
        [data-testid="stFileUploader"] {
            background: #f8fafc;
            border: 1.5px dashed #94a3b8;
            border-radius: 12px;
            padding: 6px;
        }
        div[data-baseweb="radio"] label {
            font-weight: 500;
        }
        .stTextInput input, .stNumberInput input {
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            background-color: #ffffff !important;
            color: #1e293b !important;
        }

        /* Tombol utama */
        .stButton > button {
            background: linear-gradient(135deg, #0ea5e9, #1e3a8a);
            color: #ffffff !important;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            font-size: 14.5px;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.35);
            transition: all 0.15s ease-in-out;
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(14, 165, 233, 0.45);
            color: #ffffff !important;
        }
        .stDownloadButton > button {
            background: #ffffff;
            color: #0f172a !important;
            border: 1.5px solid #0f172a;
            border-radius: 10px;
            font-weight: 600;
            width: 100%;
        }
        .stDownloadButton > button:hover {
            background: #0f172a;
            color: #ffffff !important;
        }

        /* -----------------------------------------------------------------
           6. EMPTY STATE
        ----------------------------------------------------------------- */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 70px 20px;
            text-align: center;
            border: 2px dashed #cbd5e1;
            border-radius: 16px;
            background: #f8fafc;
        }
        .empty-state .icon-circle {
            width: 72px; height: 72px;
            border-radius: 50%;
            background: linear-gradient(135deg, #e0f2fe, #dbeafe);
            display: flex; align-items: center; justify-content: center;
            font-size: 34px;
            margin-bottom: 16px;
        }
        .empty-state h4 {
            color: #334155 !important;
            font-size: 16px;
            margin: 0 0 4px 0;
        }
        .empty-state p {
            color: #94a3b8 !important;
            font-size: 13px;
            max-width: 280px;
        }

        /* -----------------------------------------------------------------
           7. DIAGNOSIS RESULT BOX
        ----------------------------------------------------------------- */
        .diagnosis-box {
            border-radius: 14px;
            padding: 22px 24px;
            margin-bottom: 18px;
            border-left: 6px solid;
        }
        .diagnosis-box.negative {
            background: #f0fdf4;
            border-color: #22c55e;
        }
        .diagnosis-box.scanty {
            background: #fffbeb;
            border-color: #f59e0b;
        }
        .diagnosis-box.positive {
            background: #fef2f2;
            border-color: #ef4444;
        }
        .diagnosis-title {
            font-size: 19px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .diagnosis-box.negative .diagnosis-title { color: #15803d !important; }
        .diagnosis-box.scanty .diagnosis-title { color: #b45309 !important; }
        .diagnosis-box.positive .diagnosis-title { color: #b91c1c !important; }
        .diagnosis-desc {
            font-size: 13.5px;
            color: #334155 !important;
            line-height: 1.6;
        }

        /* Metric mini cards */
        .metric-mini {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px 16px;
            text-align: center;
        }
        .metric-mini .val {
            font-size: 24px;
            font-weight: 800;
            color: #0f172a !important;
        }
        .metric-mini .lbl {
            font-size: 11.5px;
            color: #64748b !important;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        /* -----------------------------------------------------------------
           8. CUSTOM PROGRESS BAR (Confidence Score)
        ----------------------------------------------------------------- */
        .progress-wrap {
            margin-top: 8px;
        }
        .progress-label-row {
            display: flex;
            justify-content: space-between;
            font-size: 12.5px;
            font-weight: 600;
            color: #334155 !important;
            margin-bottom: 6px;
        }
        .progress-track {
            width: 100%;
            height: 16px;
            background: #e2e8f0;
            border-radius: 999px;
            overflow: hidden;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.08);
        }
        .progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #0ea5e9, #1e3a8a);
            transition: width 1s ease-in-out;
            box-shadow: 0 0 8px rgba(14, 165, 233, 0.5);
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #e2e8f0;
            padding: 4px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #475569 !important;
            font-weight: 600;
            font-size: 13.5px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }

        .img-caption {
            text-align: center;
            font-size: 12.5px;
            font-weight: 600;
            color: #64748b !important;
            margin-top: 6px;
        }

        hr {
            border-color: #e2e8f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 3. MODEL LOADING (CACHED)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not ULTRALYTICS_AVAILABLE:
        return None
    if not os.path.exists(path):
        return None
    return YOLO(path)


# =============================================================================
# 4. SESSION STATE INIT
# =============================================================================
def init_session_state():
    defaults = {
        "input_image": None,
        "result_image": None,
        "bta_count": None,
        "avg_confidence": None,
        "scan_done": False,
        "detections": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =============================================================================
# 5. SIDEBAR — METADATA PASIEN & PANDUAN KUALITAS CITRA
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-icon">🧬</div>
                <div class="sidebar-brand-text">
                    <b>SputumAI</b><br><span>Workspace v1.0</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">Data Pasien</div>', unsafe_allow_html=True)
        rm_number = st.text_input("Nomor Rekam Medis", placeholder="Contoh: RM-00231458")
        patient_age = st.number_input("Usia Pasien (tahun)", min_value=0, max_value=120, value=30, step=1)
        patient_gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        examiner_name = st.text_input("Nama Analis / Petugas Lab", placeholder="Contoh: dr. Andi Wijaya, Sp.PK")

        st.markdown('<div class="sidebar-section-label">Panduan Kualitas Citra</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="guidance-box">
                <b>📌 Standar Preparat Mikroskopis</b>
                <ul>
                    <li>Gunakan pembesaran objektif <b>100x</b> (minyak imersi).</li>
                    <li>Pastikan pencahayaan merata & tidak overexposed.</li>
                    <li>Fokus tajam pada lapang pandang BTA.</li>
                    <li>Hindari citra buram (motion blur).</li>
                    <li>Format file: <b>JPG / PNG</b>, resolusi ≥ 640px.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("© 2026 SputumAI Workspace — Clinical Decision Support Tool. Bukan pengganti diagnosis dokter Sp.PK.")

    return {
        "rm_number": rm_number if rm_number else "-",
        "age": patient_age,
        "gender": patient_gender,
        "examiner": examiner_name if examiner_name else "-",
    }


# =============================================================================
# 6. HEADER
# =============================================================================
def render_header():
    st.markdown(
        """
        <div class="clinical-header">
            <h1>🩺 SputumAI Workspace</h1>
            <p>Sistem Bantu Diagnosis Berbasis Kecerdasan Buatan (YOLOv8) untuk Deteksi dan Kuantifikasi
            Bakteri Tahan Asam (BTA) / <i>Mycobacterium tuberculosis</i> pada Citra Sediaan Dahak Mikroskopis.</p>
            <div class="badge-row">
                <span class="header-badge">⚙️ Model: YOLOv8</span>
                <span class="header-badge">🔬 Mode: Klinis</span>
                <span class="header-badge">🟢 Status: Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 7. HELPER — INTERPRETASI KLINIS (STANDAR SEMI-KUANTITATIF WHO/IUATLD)
# =============================================================================
def get_diagnosis_class(count: int):
    """
    Mengembalikan (kategori, css_class, label, deskripsi) berdasarkan jumlah BTA.
    Skala disederhanakan mengikuti prinsip skala IUATLD:
      0        -> Negatif
      1 - 9    -> Scanty (positif rendah)
      > 9      -> Positif
    """
    if count == 0:
        return (
            "negative",
            "negative",
            "NEGATIF (Tidak Ditemukan BTA)",
            "Tidak ditemukan Basil Tahan Asam pada seluruh lapang pandang yang dianalisis oleh sistem AI. "
            "Disarankan pemeriksaan ulang pada 2-3 sampel dahak (Sewaktu-Pagi-Sewaktu) sesuai protokol "
            "untuk konfirmasi klinis lebih lanjut.",
        )
    elif 1 <= count <= 9:
        return (
            "scanty",
            "scanty",
            "SCANTY (BTA Positif Rendah / 1-9 per LP)",
            "Ditemukan sejumlah kecil Basil Tahan Asam. Hasil ini termasuk kategori Scanty menurut skala "
            "semi-kuantitatif. Direkomendasikan korelasi dengan gejala klinis dan pemeriksaan penunjang "
            "lanjutan (mis. TCM/GeneXpert) untuk konfirmasi diagnosis.",
        )
    else:
        return (
            "positive",
            "positive",
            "POSITIF (BTA Ditemukan > 9 per Lapang Pandang)",
            "Ditemukan jumlah Basil Tahan Asam yang signifikan, mengindikasikan kemungkinan tinggi infeksi "
            "aktif Mycobacterium tuberculosis. Segera rujuk pasien untuk evaluasi klinis lanjutan oleh "
            "Dokter Spesialis Paru / Sp.PK dan pertimbangkan inisiasi tata laksana sesuai pedoman nasional TBC.",
        )


def render_progress_bar(label: str, percentage: float):
    percentage = max(0, min(100, percentage))
    st.markdown(
        f"""
        <div class="progress-wrap">
            <div class="progress-label-row">
                <span>{label}</span>
                <span>{percentage:.1f}%</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{percentage:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 8. INFERENCE
# =============================================================================
def run_inference(model, image: Image.Image):
    """Menjalankan prediksi YOLOv8 dan mengembalikan (result_image, count, avg_conf, detections)."""
    results = model.predict(source=image, conf=0.1, imgsz=640, verbose=False)
    result = results[0]

    boxes = result.boxes
    count = len(boxes) if boxes is not None else 0

    confidences = []
    if boxes is not None and count > 0:
        confidences = [float(c) for c in boxes.conf.tolist()]

    avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else 0.0

    # Gambar hasil anotasi (BGR -> RGB -> PIL)
    annotated_array = result.plot()
    annotated_image = Image.fromarray(annotated_array[:, :, ::-1])

    return annotated_image, count, avg_conf, confidences


# =============================================================================
# 9. PDF REPORT GENERATION
# =============================================================================
def generate_pdf_report(patient_info: dict, result_image: Image.Image, count: int, avg_conf: float):
    """
    Membuat laporan medis PDF. Menggunakan tempfile.NamedTemporaryFile agar gambar
    ditulis & ditutup sepenuhnya sebelum dibaca oleh FPDF, lalu file temporary dihapus.
    """
    category, css_class, label, description = get_diagnosis_class(count)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---------- Header Klinik ----------
    pdf.set_fill_color(15, 23, 42)  # navy
    pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_xy(10, 7)
    pdf.cell(0, 8, "SputumAI Workspace", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(10)
    pdf.cell(0, 6, "Laporan Hasil Pemeriksaan BTA Berbasis Kecerdasan Buatan (YOLOv8)", ln=1)

    pdf.set_text_color(30, 41, 59)
    pdf.ln(14)

    # ---------- Data Pasien ----------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Data Pasien & Pemeriksaan", ln=1)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10.5)
    exam_time = datetime.now().strftime("%d %B %Y, %H:%M:%S")

    info_rows = [
        ("Nomor Rekam Medis", patient_info.get("rm_number", "-")),
        ("Usia Pasien", f"{patient_info.get('age', '-')} tahun"),
        ("Jenis Kelamin", patient_info.get("gender", "-")),
        ("Analis / Petugas", patient_info.get("examiner", "-")),
        ("Waktu Pemeriksaan", exam_time),
    ]
    for label_txt, value_txt in info_rows:
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.cell(55, 7, f"{label_txt}", border=0)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.cell(0, 7, f": {value_txt}", ln=1)

    pdf.ln(4)

    # ---------- Gambar Hasil Deteksi (via tempfile) ----------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Citra Hasil Deteksi AI", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    tmp_path = None
    try:
        # Buat file temporary, tulis gambar, lalu PASTIKAN ditutup sebelum dibaca FPDF
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp_file.name
        tmp_file.close()  # tutup handle agar bisa ditulis ulang oleh PIL secara aman

        rgb_image = result_image.convert("RGB")
        rgb_image.save(tmp_path, format="PNG")

        # Hitung ukuran gambar proporsional agar muat di halaman (max width 130mm)
        img_w_px, img_h_px = rgb_image.size
        max_width_mm = 130
        ratio = max_width_mm / img_w_px
        display_h = img_h_px * ratio

        x_center = (210 - max_width_mm) / 2
        pdf.image(tmp_path, x=x_center, y=pdf.get_y(), w=max_width_mm, h=display_h)
        pdf.set_y(pdf.get_y() + display_h + 6)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ---------- Hasil Kuantitatif ----------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Hasil Kuantitatif", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(65, 7, "Jumlah BTA Terdeteksi", border=0)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.cell(0, 7, f": {count} basil", ln=1)

    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(65, 7, "Rata-rata Confidence Score", border=0)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.cell(0, 7, f": {avg_conf:.1f}%", ln=1)

    pdf.ln(4)

    # ---------- Interpretasi Medis ----------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Interpretasi Medis (Kesimpulan)", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    color_map = {
        "negative": (34, 197, 94),
        "scanty": (245, 158, 11),
        "positive": (239, 68, 68),
    }
    r, g, b = color_map.get(css_class, (100, 116, 139))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 9, f"  {label}", ln=1, fill=True)

    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 6, description)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: Laporan ini dihasilkan oleh sistem bantu diagnosis berbasis kecerdasan buatan (Clinical "
        "Decision Support System) dan WAJIB dikonfirmasi serta divalidasi oleh Dokter Spesialis Patologi Klinik "
        "(Sp.PK) atau tenaga medis berwenang sebelum digunakan sebagai dasar diagnosis final."
    )

    return bytes(pdf.output(dest="S"))


# =============================================================================
# 10. MAIN AREA & HASIL ANALISIS (DIGABUNG AGAR LANGSUNG MUNCUL)
# =============================================================================
def render_main_area(model, patient_info: dict):
    col_input, col_workspace = st.columns([1, 1.4], gap="medium")

    # --------------------------- KOLOM INPUT --------------------------------
    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📥 Sumber Citra</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Pilih metode input citra sediaan dahak mikroskopis</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Metode Input",
            ["Unggah File (Drag & Drop)", "Kamera Mikroskop"],
            label_visibility="collapsed",
        )

        uploaded_image = None
        if input_mode == "Unggah File (Drag & Drop)":
            uploaded_file = st.file_uploader(
                "Seret & lepas citra di sini, atau klik untuk memilih file",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
        else:
            camera_file = st.camera_input("Ambil citra langsung dari kamera mikroskop")
            if camera_file is not None:
                uploaded_image = Image.open(camera_file).convert("RGB")

        if uploaded_image is not None:
            st.session_state.input_image = uploaded_image

        st.markdown("</div>", unsafe_allow_html=True)

        if not ULTRALYTICS_AVAILABLE:
            st.warning("⚠️ Modul `ultralytics` belum terpasang.")
        elif model is None:
            st.warning(f"⚠️ Model `{MODEL_PATH}` tidak ditemukan.")

    # ------------------------- KOLOM WORKSPACE -------------------------------
    with col_workspace:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔬 Ruang Kerja Analisis</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Pratinjau citra dan eksekusi pemindaian AI</div>', unsafe_allow_html=True)

        if st.session_state.input_image is None:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="icon-circle">🩺</div>
                    <h4>Belum Ada Citra Dimuat</h4>
                    <p>Unggah atau ambil citra sediaan dahak melalui panel di sebelah kiri untuk memulai analisis.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.image(st.session_state.input_image, use_container_width=True, caption="Citra Input (Original)")
            run_scan = st.button("🚀 Jalankan Pemindaian AI", use_container_width=True)

            if run_scan:
                if model is None:
                    st.error("Model AI tidak tersedia.")
                else:
                    with st.spinner("Menganalisis citra mikroskopis... AI sedang mendeteksi BTA."):
                        annotated_image, count, avg_conf, confidences = run_inference(
                            model, st.session_state.input_image
                        )
                    st.session_state.result_image = annotated_image
                    st.session_state.bta_count = count
                    st.session_state.avg_confidence = avg_conf
                    st.session_state.detections = confidences
                    st.session_state.scan_done = True
                    st.toast("✅ Pemindaian AI selesai!", icon="✅")

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # BAGIAN HASIL (MUNCUL OTOMATIS TEPAT DI BAWAH KETIKA SCAN SELESAI)
    # =========================================================================
    if st.session_state.scan_done and st.session_state.result_image is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Hasil Pemeriksaan & Laporan Klinis</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Tinjau hasil deteksi visual dan laporan interpretasi klinis</div>', unsafe_allow_html=True)

        tab_visual, tab_report = st.tabs(["🖼️ Analisis Visual", "📋 Laporan Klinis"])

        count = st.session_state.bta_count
        avg_conf = st.session_state.avg_confidence
        category, css_class, label, description = get_diagnosis_class(count)

        # TAB 1: ANALISIS VISUAL
        with tab_visual:
            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                st.image(st.session_state.input_image, use_container_width=True)
                st.markdown('<div class="img-caption">Citra Asli (Original)</div>', unsafe_allow_html=True)
            with col_b:
                st.image(st.session_state.result_image, use_container_width=True)
                st.markdown('<div class="img-caption">Hasil Deteksi AI (Annotated)</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-mini"><div class="val">{count}</div><div class="lbl">Total BTA Terdeteksi</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-mini"><div class="val">{avg_conf:.1f}%</div><div class="lbl">Rata-rata Confidence</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-mini"><div class="val">{category.upper()}</div><div class="lbl">Kategori Sementara</div></div>', unsafe_allow_html=True)

        # TAB 2: LAPORAN KLINIS
        with tab_report:
            st.markdown(
                f"""
                <div class="diagnosis-box {css_class}">
                    <div class="diagnosis-title">{label}</div>
                    <div class="diagnosis-desc">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Visualisasi Confidence Score Model AI")
            render_progress_bar("Rata-rata Confidence Score Deteksi", avg_conf)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Ringkasan Data Pemeriksaan")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**Nomor Rekam Medis:** {patient_info['rm_number']}")
                st.markdown(f"**Usia Pasien:** {patient_info['age']} tahun")
            with info_col2:
                st.markdown(f"**Jenis Kelamin:** {patient_info['gender']}")
                st.markdown(f"**Waktu Analisis:** {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Tombol Unduh Laporan PDF
            pdf_bytes = generate_pdf_report(patient_info, st.session_state.result_image, count, avg_conf)
            file_name = f"Laporan_SputumAI_{patient_info['rm_number'].replace(' ', '_')}.pdf"

            st.download_button(
                label="⬇️ Unduh Laporan Medis (PDF)",
                data=pdf_bytes,
                file_name=file_name,
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# 11. HASIL — TABS (ANALISIS VISUAL & LAPORAN KLINIS)
# =============================================================================
def render_results_section(patient_info: dict):
    if not st.session_state.scan_done or st.session_state.result_image is None:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Hasil Pemeriksaan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Tinjau hasil deteksi visual dan laporan interpretasi klinis</div>',
        unsafe_allow_html=True,
    )

    tab_visual, tab_report = st.tabs(["🖼️ Analisis Visual", "📋 Laporan Klinis"])

    count = st.session_state.bta_count
    avg_conf = st.session_state.avg_confidence
    category, css_class, label, description = get_diagnosis_class(count)

    # ------------------------- TAB 1: ANALISIS VISUAL -------------------------
    with tab_visual:
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.image(st.session_state.input_image, use_container_width=True)
            st.markdown('<div class="img-caption">Citra Asli (Original)</div>', unsafe_allow_html=True)
        with col_b:
            st.image(st.session_state.result_image, use_container_width=True)
            st.markdown('<div class="img-caption">Hasil Deteksi AI (Annotated)</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-mini"><div class="val">{count}</div><div class="lbl">Total BTA Terdeteksi</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-mini"><div class="val">{avg_conf:.1f}%</div><div class="lbl">Rata-rata Confidence</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-mini"><div class="val">{category.upper()}</div><div class="lbl">Kategori Sementara</div></div>',
                unsafe_allow_html=True,
            )

    # ------------------------- TAB 2: LAPORAN KLINIS ---------------------------
    with tab_report:
        st.markdown(
            f"""
            <div class="diagnosis-box {css_class}">
                <div class="diagnosis-title">{label}</div>
                <div class="diagnosis-desc">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Visualisasi Confidence Score Model AI")
        render_progress_bar("Rata-rata Confidence Score Deteksi", avg_conf)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Ringkasan Data Pemeriksaan")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"**Nomor Rekam Medis:** {patient_info['rm_number']}")
            st.markdown(f"**Usia Pasien:** {patient_info['age']} tahun")
        with info_col2:
            st.markdown(f"**Jenis Kelamin:** {patient_info['gender']}")
            st.markdown(f"**Waktu Analisis:** {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------- Tombol Unduh Laporan PDF ----------------------
        pdf_bytes = generate_pdf_report(patient_info, st.session_state.result_image, count, avg_conf)
        file_name = f"Laporan_SputumAI_{patient_info['rm_number'].replace(' ', '_')}.pdf"

        st.download_button(
            label="⬇️ Unduh Laporan Medis (PDF)",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 12. MAIN APP ENTRYPOINT
# =============================================================================
def main():
    inject_custom_css()
    init_session_state()

    patient_info = render_sidebar()
    render_header()

    model = load_model(MODEL_PATH)

    render_main_area(model, patient_info)
    render_results_section(patient_info)


if __name__ == "__main__":
    main()
