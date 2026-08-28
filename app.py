# =============================================================================
# SPUTUM-AI : END-TO-END TUBERCULOSIS PLATFORM (FULL CLINICAL & AI WORKSPACE)
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
# 1. KONFIGURASI HALAMAN & CSS PRO
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
        .card-title { font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 12px; }

        [data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.8rem !important; font-weight: 800 !important; }
        [data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 600 !important; }
        
        .diagnosis-box { border-radius: 12px; padding: 18px; margin-bottom: 15px; border-left: 6px solid; }
        .diagnosis-box.negative { background: #f0fdf4; border-color: #22c55e; color: #15803d; }
        .diagnosis-box.scanty { background: #fffbeb; border-color: #f59e0b; color: #b45309; }
        .diagnosis-box.positive { background: #fef2f2; border-color: #ef4444; color: #b91c1c; }
        
        .progress-track { width: 100%; height: 12px; background: #e2e8f0; border-radius: 999px; overflow: hidden; margin-top: 6px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #0ea5e9, #1e3a8a); }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    path = "best.pt"
    if not ULTRALYTICS_AVAILABLE or not os.path.exists(path): return None
    return YOLO(path)

inject_custom_css()
model = load_model()

# =============================================================================
# 2. NAVIGASI SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🏥 SputumAI Ecosystem")
    st.caption("Platform Penanganan TBC Terpadu")
    st.markdown("---")
    menu = st.radio("Pilih Modul Navigasi:", [
        "📊 Dashboard Epidemiologi",
        "🔬 Workspace AI (Deteksi & PDF)", 
        "🗺️ Peta Rujukan Faskes", 
        "📚 Pusat Edukasi & Kuis"
    ])
    st.markdown("---")
    st.info("🔒 Sistem terenkripsi untuk menjaga privasi pasien sesuai standar etika medis.")

# =============================================================================
# 3. HELPER FUNGSI KLINIS & PDF
# =============================================================================
def get_diagnosis_class(count: int):
    if count == 0:
        return ("negative", "NEGATIF (Tidak Ditemukan BTA)", "Tidak ditemukan Basil Tahan Asam pada lapang pandang ini.")
    elif 1 <= count <= 9:
        return ("scanty", "SCANTY (BTA Positif Rendah / 1-9 per LP)", "Ditemukan jumlah kecil BTA. Perlu korelasi klinis lanjutan.")
    else:
        return ("positive", "POSITIF (BTA Ditemukan > 9 per Lapang Pandang)", "Ditemukan BTA signifikan. Segera rujuk untuk penanganan spesialis paru.")

def generate_pdf_report(patient_info: dict, result_image: Image.Image, count: int, avg_conf: float):
    _, label, description = get_diagnosis_class(count)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(10, 7)
    pdf.cell(0, 8, "SputumAI Workspace", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(10)
    pdf.cell(0, 6, "Laporan Hasil Pemeriksaan BTA Berbasis AI", ln=1)

    pdf.set_text_color(30, 41, 59)
    pdf.ln(14)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Data Pasien & Pemeriksaan", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    exam_time = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    info_rows = [
        ("Nomor Rekam Medis", patient_info.get("rm_number", "-")),
        ("Usia Pasien", f"{patient_info.get('age', '-')} tahun"),
        ("Jenis Kelamin", patient_info.get("gender", "-")),
        ("Analis / Petugas", patient_info.get("examiner", "-")),
        ("Waktu Pemeriksaan", exam_time),
    ]
    for k, v in info_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 6, k, border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f": {v}", ln=1)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Citra Hasil Deteksi AI", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    tmp_path = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_path = tmp_file.name
        tmp_file.close()
        result_image.convert("RGB").save(tmp_path, format="PNG")
        pdf.image(tmp_path, x=40, y=pdf.get_y(), w=130)
        pdf.set_y(pdf.get_y() + 95)
    finally:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Hasil Kuantitatif & Interpretasi", ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.cell(50, 6, "Total BTA Terdeteksi", border=0)
    pdf.cell(0, 6, f": {count} basil", ln=1)
    pdf.cell(50, 6, "Confidence Score", border=0)
    pdf.cell(0, 6, f": {avg_conf:.1f}%", ln=1)
    pdf.cell(50, 6, "Kategori", border=0)
    pdf.cell(0, 6, f": {label}", ln=1)
    pdf.ln(2)
    pdf.multi_cell(0, 5, f"Interpretasi: {description}")

    pdf_output = pdf.output(dest="S")
    return pdf_output.encode("latin-1") if isinstance(pdf_output, str) else bytes(pdf_output)

# =============================================================================
# MODUL 1: DASHBOARD EPIDEMIOLOGI
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
        st.metric(label="Zona Risiko Tinggi", value="Jabar & Jatim", delta="Prioritas Rujukan")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Tren Insidensi Kasus TBC (2019 - 2023)")
    chart_data = pd.DataFrame(
        {"Tahun": ["2019", "2020", "2021", "2022", "2023"], "Jumlah Kasus Ditemukan": [568000, 393000, 397000, 717000, 809000]}
    ).set_index("Tahun")
    st.bar_chart(chart_data, color="#0ea5e9")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MODUL 2: WORKSPACE AI (DIAGNOSTIK & PDF REPORT)
# =============================================================================
elif menu == "🔬 Workspace AI (Deteksi & PDF)":
    st.markdown("""
    <div class="hero-banner">
        <div><h1>Workspace AI <span style="color: #38bdf8;">Diagnostik Klinis</span></h1>
        <p>Analisis citra sediaan dahak mikroskopis menggunakan YOLOv8 lengkap dengan Laporan PDF.</p></div>
        <div style="font-size: 2.5rem;">🔬</div>
    </div>
    """, unsafe_allow_html=True)

    if "scan_done" not in st.session_state: st.session_state.scan_done = False

    col_meta, col_input, col_workspace = st.columns([1, 1, 1.3], gap="medium")

    with col_meta:
        st.markdown('<div class="card"><div class="card-title">📋 Metadata Pasien</div>', unsafe_allow_html=True)
        rm_number = st.text_input("No. Rekam Medis", value="RM-882145")
        patient_age = st.number_input("Usia Pasien", value=30, min_value=1, max_value=120)
        patient_gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        examiner_name = st.text_input("Nama Analis / Dokter", value="dr. Andi, Sp.PK")
        st.markdown('</div>', unsafe_allow_html=True)
        patient_info = {"rm_number": rm_number, "age": patient_age, "gender": patient_gender, "examiner": examiner_name}

    with col_input:
        st.markdown('<div class="card"><div class="card-title">📥 Sumber Citra</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Unggah Sediaan Dahak", type=["jpg", "jpeg", "png"])
        uploaded_image = None
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.session_state.input_image = uploaded_image
            st.session_state.scan_done = False
        st.markdown('</div>', unsafe_allow_html=True)

    with col_workspace:
        st.markdown('<div class="card"><div class="card-title">🖥️ Ruang Eksekusi AI</div>', unsafe_allow_html=True)
        if "input_image" not in st.session_state or st.session_state.input_image is None:
            st.info("Silakan unggah citra di panel tengah.")
        else:
            st.image(st.session_state.input_image, use_container_width=True, caption="Citra Input Asli")
            if st.button("🚀 Jalankan Pemindaian AI", use_container_width=True):
                if model is None:
                    st.error("Model `best.pt` tidak ditemukan.")
                else:
                    with st.spinner("AI sedang mendeteksi BTA..."):
                        results = model.predict(source=st.session_state.input_image, conf=0.1, imgsz=640, verbose=False)[0]
                        boxes = results.boxes
                        count = len(boxes) if boxes is not None else 0
                        confidences = [float(c) for c in boxes.conf.tolist()] if boxes is not None and count > 0 else []
                        avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else 0.0
                        
                        annotated_array = results.plot()
                        st.session_state.result_image = Image.fromarray(annotated_array[:, :, ::-1])
                        st.session_state.bta_count = count
                        st.session_state.avg_confidence = avg_conf
                        st.session_state.scan_done = True
                        st.toast("✅ Pemindaian selesai!", icon="✅")
        st.markdown('</div>', unsafe_allow_html=True)

    # Hasil Deteksi & Laporan
    if st.session_state.get("scan_done", False) and st.session_state.get("result_image") is not None:
        st.markdown('<div class="card"><div class="card-title">📊 Hasil Pemeriksaan & Laporan Klinis</div>', unsafe_allow_html=True)
        tab_v, tab_r = st.tabs(["🖼️ Analisis Visual", "📋 Laporan & Unduh PDF"])

        count = st.session_state.bta_count
        avg_conf = st.session_state.avg_confidence
        css_class, label, desc = get_diagnosis_class(count)

        with tab_v:
            ca, cb = st.columns(2)
            with ca: st.image(st.session_state.input_image, use_container_width=True, caption="Original")
            with cb: st.image(st.session_state.result_image, use_container_width=True, caption="AI Annotated")
            
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Total BTA Terdeteksi", f"{count} Sel")
            with m2: st.metric("Rata-rata Confidence", f"{avg_conf:.1f}%")
            with m3: st.metric("Kategori Status", label.split()[0])

        with tab_r:
            st.markdown(f"""
                <div class="diagnosis-box {css_class}">
                    <h3 style="margin:0 0 5px 0;">{label}</h3>
                    <p style="margin:0;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

            st.write("**Visualisasi Confidence Score:**")
            st.markdown(f"""
                <div style="display:flex; justify-content:space-between; font-weight:600; font-size:13px;">
                    <span>Tingkat Kepercayaan Model</span><span>{avg_conf:.1f}%</span>
                </div>
                <div class="progress-track"><div class="progress-fill" style="width:{avg_conf}%;"></div></div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            pdf_bytes = generate_pdf_report(patient_info, st.session_state.result_image, count, avg_conf)
            st.download_button(
                label="⬇️ Unduh Dokumen Laporan Medis (PDF)",
                data=pdf_bytes,
                file_name=f"SputumAI_Report_{patient_info['rm_number']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_final"
            )
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MODUL 3: PETA RUJUKAN FASKES
# =============================================================================
elif menu == "🗺️ Peta Rujukan Faskes":
    import math

    st.markdown("""
    <div class="hero-banner">
        <div><h1>Smart National Referral & <span style="color: #38bdf8;">Peta Faskes Indonesia</span></h1>
        <p>Direktori Nasional Rumah Sakit Rujukan TBC & Dokter Spesialis Terdekat Berdasarkan Lokasi Anda.</p></div>
        <div style="font-size: 2.5rem;">🗺️</div>
    </div>
    """, unsafe_allow_html=True)

    # Simulasi Titik Koordinat Pasien (Default: Malang Pusat, bisa diintegrasikan GPS Browser)
    st.markdown("### 📍 Pengaturan Lokasi Anda")
    col_loc1, col_loc2 = st.columns(2)
    with col_loc1:
        user_lat = st.number_input("Latitude Anda", value=-7.9666, format="%.4f")
    with col_loc2:
        user_lon = st.number_input("Longitude Anda", value=112.6326, format="%.4f")

    st.info("💡 **Sistem Otomatis:** Daftar Rumah Sakit di bawah ini diurutkan secara real-time mulai dari **jarak terdekat hingga terjauh** dari posisi koordinat Anda.")

    # 1. Formula Haversine untuk menghitung jarak (KM)
    def calculate_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Radius bumi dalam kilometer
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    # 2. Database Nasional Rumah Sakit & Dokter (Sampel Lintas Kota di Indonesia)
    hospitals_db = [
        {
            "name": "RSUD Dr. Saiful Anwar (RSSA)", "city": "Malang", "lat": -7.9723, "lon": 112.6300, "color": "red", "icon": "hospital-o",
            "doctor": "dr. Susilo, Sp.P(K)", "phone": "0341362101", "facilities": "TBC RO, GeneXpert, Poli DOTS",
            "address": "Jl. Jaksa Agung Suprapto No.2, Malang"
        },
        {
            "name": "RS Paru dr. Ario Wirawan", "city": "Salatiga", "lat": -7.3311, "lon": 110.5083, "color": "red", "icon": "hospital-o",
            "doctor": "dr. Hendra, Sp.P", "phone": "0298326088", "facilities": "Pusat Rujukan Paru Nasional, Isolasi",
            "address": "Jl. Hasanuddin No.80, Salatiga"
        },
        {
            "name": "RSUP Persahabatan", "city": "Jakarta Timur", "lat": -6.1954, "lon": 106.8923, "color": "red", "icon": "hospital-o",
            "doctor": "Prof. Dr. dr. Faisal Yunus, Sp.P(K)", "phone": "0214891708", "facilities": "Pusat Respirasi Nasional, Lab Biosafety",
            "address": "Jl. Persahabatan Raya, Jakarta Timur"
        },
        {
            "name": "RSUD Dr. Soetomo", "city": "Surabaya", "lat": -7.2658, "lon": 112.7554, "color": "red", "icon": "hospital-o",
            "doctor": "dr. Retno Asih, Sp.P", "phone": "0315501011", "facilities": "TBC RO Tingkat Lanjut, GeneXpert",
            "address": "Jl. Mayjen Prof. Dr. Moestopo No.6-8, Surabaya"
        },
        {
            "name": "Klinik Paru Medika", "city": "Malang", "lat": -7.9555, "lon": 112.6150, "color": "blue", "icon": "user-md",
            "doctor": "dr. Anita, Sp.P", "phone": "0341471234", "facilities": "Skrining Awal & Konsultasi",
            "address": "Jl. Letjen Sutoyo No.45, Malang"
        }
    ]

    # 3. Hitung jarak untuk setiap faskes dan masukkan ke atribut data
    for h in hospitals_db:
        h["distance"] = calculate_distance(user_lat, user_lon, h["lat"], h["lon"])

    # 4. Urutkan berdasarkan jarak terdekat (ascending)
    sorted_hospitals = sorted(hospitals_db, key=lambda x: x["distance"])

    # 5. Render Peta Berpusat di Lokasi Pasien
    m = folium.Map(location=[user_lat, user_lon], zoom_start=11)

    # Tambahkan Marker Lokasi Pasien
    folium.Marker(
        location=[user_lat, user_lon],
        popup="<b>Lokasi Anda Saat Ini</b>",
        tooltip="Titik Koordinat Anda",
        icon=folium.Icon(color="green", icon="user", prefix='fa')
    ).add_to(m)

    # Render Rumah Sakit ke Peta & List
    col_map, col_list = st.columns([1.5, 1])

    with col_map:
        for h in sorted_hospitals:
            html_content = f"""
                <div style="font-family: 'Segoe UI', sans-serif; width: 280px; font-size: 13px; color: #1e293b;">
                    <b style="color: #0f172a; font-size: 14px;">🏥 {h['name']}</b><hr style="margin:5px 0;">
                    <b>Kota:</b> {h['city']}<br>
                    <b>Jarak:</b> ~{h['distance']:.2f} km dari Anda<br>
                    <b>Alamat:</b> {h['address']}<br>
                    <b>Spesialis:</b> {h['doctor']}<br>
                    <b>Layanan:</b> {h['facilities']}<br>
                    <b>Telepon:</b> {h['phone']}<br><br>
                    <a href='tel:{h['phone']}' target='_blank' style='background:#22c55e; color:white; padding:6px 12px; border-radius:6px; text-decoration:none; font-weight:600; display:inline-block;'>📞 Hubungi Rumah Sakit</a>
                </div>
            """
            iframe = folium.IFrame(html=html_content, width=290, height=215)
            popup = folium.Popup(iframe, max_width=290)
            
            folium.Marker(
                location=[h["lat"], h["lon"]], popup=popup, tooltip=f"{h['name']} ({h['distance']:.1f} km)",
                icon=folium.Icon(color=h["color"], icon=h["icon"], prefix='fa')
            ).add_to(m)
        
        st_folium(m, width="100%", height=450)

    with col_list:
        st.markdown("#### 📋 Urutan Faskes Terdekat")
        for i, h in enumerate(sorted_hospitals, 1):
            st.markdown(f"""
                <div class="card" style="padding: 12px; margin-bottom: 10px;">
                    <b style="color: #0ea5e9;">#{i} {h['name']}</b><br>
                    <span style="font-size: 12px; color: #64748b;">📍 {h['city']} — <b>~{h['distance']:.2f} km</b></span><br>
                    <span style="font-size: 12px;">👨‍⚕️ {h['doctor']}</span><br>
                    <a href="tel:{h['phone']}" style="font-size: 12px; text-decoration:none;">📞 {h['phone']}</a>
                </div>
            """, unsafe_allow_html=True)
# =============================================================================
# MODUL 4: PUSAT EDUKASI & KUIS
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
                
