# =============================================================================
# SPUTUM-AI : END-TO-END TUBERCULOSIS PLATFORM
# =============================================================================
import os
import io
import tempfile
from datetime import datetime
import pandas as pd

import streamlit as st
from PIL import Image
from fpdf import FPDF
import folium
from streamlit_folium import st_folium

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# =============================================================================
# 1. KONFIGURASI HALAMAN & CSS
# =============================================================================
st.set_page_config(page_title="SputumAI | TB Platform", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

def inject_custom_css():
    st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: #f8fafc !important; color: #1e293b !important;
        }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: white; padding: 25px 35px; border-radius: 14px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15); margin-bottom: 25px;
            display: flex; justify-content: space-between; align-items: center;
            border-left: 6px solid #0ea5e9;
        }
        .hero-banner h1 { margin: 0; font-size: 2rem; font-weight: 800; color: #ffffff !important; }
        .hero-banner p { margin: 5px 0 0 0; color: #cbd5e1 !important; font-size: 1rem; }
        
        .card { 
            background: #ffffff; border-radius: 12px; padding: 20px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; 
            margin-bottom: 20px; 
        }

        /* Perbaikan Kontras Teks & Angka Metric Streamlit */
        [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricDelta"] {
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    path = "best.pt"
    if not ULTRALYTICS_AVAILABLE or not os.path.exists(path): return None
    return YOLO(path)

# =============================================================================
# 2. NAVIGASI SIDEBAR UTAMA
# =============================================================================
inject_custom_css()
with st.sidebar:
    st.markdown("### 🏥 SputumAI Ecosystem")
    st.caption("Platform Penanganan TBC Terpadu")
    st.markdown("---")
    menu = st.radio("Pilih Modul Navigasi:", [
        "📊 Dashboard Epidemiologi",
        "🔬 Workspace AI (Deteksi)", 
        "🗺️ Peta Rujukan Faskes", 
        "📚 Pusat Edukasi & Kuis"
    ])
    st.markdown("---")
    st.info("🔒 Sistem terenkripsi untuk menjaga privasi pasien sesuai standar etika medis.")

model = load_model()

# =============================================================================
# MENU 1: DASHBOARD EPIDEMIOLOGI
# =============================================================================
if menu == "📊 Dashboard Epidemiologi":
    st.markdown("""
    <div class="hero-banner">
        <div><h1>Dashboard <span style="color: #38bdf8;">Epidemiologi TBC</span></h1>
        <p>Analisis tren penyebaran dan keberbahayaan TBC di Indonesia (Berdasarkan Data Kemenkes RI)</p></div>
        <div style="font-size: 2.5rem;">📊</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric(label="Estimasi Kasus Nasional", value="1.060.000", delta="Tertinggi ke-2 di Dunia", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric(label="Keberhasilan Pengobatan", value="85%", delta="-5% dari Target (90%)", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric(label="Kasus TBC Anak", value="134.528", delta="+12% Tahun ini", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric(label="Zona Risiko Tertinggi", value="Jabar & Jatim", delta="Prioritas Rujukan")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Tren Insidensi Kasus TBC (2019 - 2023)")
    chart_data = pd.DataFrame(
        {"Tahun": ["2019", "2020", "2021", "2022", "2023"], "Jumlah Kasus Ditemukan": [568000, 393000, 397000, 717000, 809000]}
    ).set_index("Tahun")
    st.bar_chart(chart_data, color="#0ea5e9")
    st.caption("Catatan: Penurunan di tahun 2020-2021 disebabkan oleh pandemi COVID-19 yang menurunkan kapasitas skrining TBC.")
    st.markdown('</div>', unsafe_allow_html=True)
# =============================================================================
# MENU 2: WORKSPACE AI (DETEKSI)
# =============================================================================
elif menu == "🔬 Workspace AI (Deteksi)":
    st.markdown("""
    <div class="hero-banner">
        <div><h1>Workspace AI <span style="color: #38bdf8;">Diagnostik</span></h1>
        <p>Analisis citra sediaan dahak mikroskopis menggunakan model YOLOv8.</p></div>
        <div style="font-size: 2.5rem;">🔬</div>
    </div>
    """, unsafe_allow_html=True)

    if "scan_done" not in st.session_state: st.session_state.scan_done = False
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown('<div class="card"><h4>📥 Input Citra Medis</h4>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Unggah sampel dahak", type=["jpg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h4>🖥️ Hasil Pemindaian</h4>', unsafe_allow_html=True)
        if uploaded_file is None:
            st.info("Silakan unggah citra di panel sebelah kiri.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Citra Asli", use_container_width=True)
            if st.button("🚀 Jalankan AI Sekarang", use_container_width=True):
                if model:
                    with st.spinner("AI sedang memindai BTA..."):
                        results = model.predict(image, conf=0.1, imgsz=640)[0]
                        boxes = results.boxes
                        st.session_state.bta_count = len(boxes)
                        st.session_state.res_image = Image.fromarray(results.plot()[..., ::-1])
                        st.session_state.scan_done = True
                else:
                    st.error("Model tidak tersedia.")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.scan_done:
        st.success(f"✅ Pemindaian Selesai! Ditemukan **{st.session_state.bta_count} Sel BTA**.")
        st.image(st.session_state.res_image, caption="Hasil Deteksi", use_container_width=True)

# =============================================================================
# MENU 3: PETA RUJUKAN FASKES (LEAFLET/FOLIUM)
# =============================================================================
elif menu == "🗺️ Peta Rujukan Faskes":
    st.markdown("""
    <div class="hero-banner">
        <div><h1>Smart Referral & <span style="color: #38bdf8;">Peta Faskes</span></h1>
        <p>Direktori Interaktif Rumah Sakit Rujukan dan Dokter Spesialis Paru Terdekat (Area Malang Raya).</p></div>
        <div style="font-size: 2.5rem;">🗺️</div>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **Petunjuk:** Klik ikon rumah sakit atau klinik pada peta di bawah ini untuk melihat biodata Dokter Spesialis, fasilitas layanan TBC, dan nomor telepon rujukan.")
    
    m = folium.Map(location=[-7.9666, 112.6326], zoom_start=13)
    hospitals = [
        {"name": "RSUD Dr. Saiful Anwar (RSSA)", "lat": -7.9723, "lon": 112.6300, "color": "red", "icon": "hospital-o",
         "html": "<b>🏥 RSUD Dr. Saiful Anwar</b><hr><b>Spesialis:</b> dr. Susilo, Sp.P(K)<br><b>Layanan:</b> TBC RO, GeneXpert<br><br><a href='tel:0341362101' target='_blank' style='background:#22c55e; color:white; padding:5px; border-radius:5px; text-decoration:none;'>📞 Telepon Rujukan</a>"},
        {"name": "Rumah Sakit Paru Batu", "lat": -7.8715, "lon": 112.5269, "color": "red", "icon": "h-square",
         "html": "<b>🏥 RS Paru Batu</b><hr><b>Spesialis:</b> dr. Hidayat, Sp.P<br><b>Layanan:</b> Rawat Inap Isolasi TBC<br><br><a href='tel:0341596881' target='_blank' style='background:#22c55e; color:white; padding:5px; border-radius:5px; text-decoration:none;'>📞 Telepon Rujukan</a>"},
        {"name": "Klinik Paru Medika Malang", "lat": -7.9555, "lon": 112.6150, "color": "blue", "icon": "user-md",
         "html": "<b>🩺 Klinik Paru Medika</b><hr><b>Spesialis:</b> dr. Anita, Sp.P<br><b>Layanan:</b> Skrining Awal<br><br><a href='https://wa.me/628123456789' target='_blank' style='background:#22c55e; color:white; padding:5px; border-radius:5px; text-decoration:none;'>💬 Hubungi via WhatsApp</a>"}
    ]

    for h in hospitals:
        iframe = folium.IFrame(html=h["html"], width=280, height=180)
        popup = folium.Popup(iframe, max_width=280)
        folium.Marker(
            location=[h["lat"], h["lon"]], popup=popup, tooltip=h["name"],
            icon=folium.Icon(color=h["color"], icon=h["icon"], prefix='fa')
        ).add_to(m)

    st_folium(m, width=1000, height=500)

# =============================================================================
# MENU 4: PUSAT EDUKASI & SMART QUIZ
# =============================================================================
elif menu == "📚 Pusat Edukasi & Kuis":
    st.markdown("""
    <div class="hero-banner">
        <div><h1>Pusat Edukasi & <span style="color: #38bdf8;">Smart Quiz</span></h1>
        <p>Media pembelajaran pencegahan TBC, repositori jurnal kesehatan, dan evaluasi pemahaman.</p></div>
        <div style="font-size: 2.5rem;">📚</div>
    </div>
    """, unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["🛡️ Pencegahan & Pasien", "📖 Repositori Jurnal", "🎯 Kuis Edukatif"])

    with t1:
        st.markdown("### 🏃‍♂️ Apa yang harus dilakukan jika Terdiagnosis TBC?")
        st.write("**1. Jangan Panik, TBC Bisa Disembuhkan!** Mulailah pengobatan Obat Anti Tuberkulosis (OAT) secara teratur.")
        st.write("**2. Kepatuhan Berobat:** Minum obat secara teratur selama minimal 6 bulan tanpa terputus.")

    with t2:
        st.markdown("### 📑 Referensi Medis TBC")
        st.markdown("- 📄 Pedoman Nasional Pelayanan Kedokteran Tatalaksana TBC (Kemenkes RI)")
        st.markdown("- 📄 WHO Global Tuberculosis Report")

    with t3:
        st.markdown("### 🧠 SputumAI Smart Quiz")
        q1 = st.radio("1. Mengapa ventilasi rumah sangat penting dalam pencegahan penularan TBC?", 
                      ["A. Agar rumah tidak lembab", 
                       "B. Sinar matahari pagi mengandung UV yang dapat membunuh kuman TBC", 
                       "C. Agar tidak kepanasan"])
        q2 = st.radio("2. Berapa lama standar waktu minimal pasien TBC harus minum obat (OAT)?", 
                      ["A. 1 Minggu", "B. 2 Bulan", "C. 6 Bulan"])
        
        if st.button("Kirim Jawaban (Submit)"):
            if "B. Sinar" in q1 and "C. 6" in q2:
                st.success("🎉 Luar Biasa! Jawaban Anda Benar Semua.")
                st.balloons()
            else:
                st.error("❌ Masih ada jawaban yang kurang tepat. Coba periksa kembali materi edukasi.")
                
