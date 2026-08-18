import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import datetime
import tempfile
from fpdf import FPDF

# 1. Konfigurasi Halaman
st.set_page_config(page_title="SputumAI | Deteksi TBC", layout="wide")

# 2. Custom CSS
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
    
    col_upload, col_instruksi = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
    with col_instruksi:
        st.info("""
        **Panduan Standar:**
        - Format didukung: JPG, JPEG, PNG
        - Pastikan pencahayaan mikroskop memadai.
        - Disarankan menggunakan sampel dengan pewarnaan standar (misal: Ziehl-Neelsen).
        """)
        
    st.markdown("---")

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
                    
                    # Variabel penampung teks untuk Cetak PDF
                    kategori_teks = ""
                    interpretasi_teks = ""
                    
                    if jumlah_bakteri == 0:
                        kategori_teks = "Negatif / Bersih"
                        interpretasi_teks = "Tidak ditemukan indikasi bakteri Mycobacterium tuberculosis pada area citra ini. Disarankan memeriksa lapang pandang sampel lainnya untuk memastikan hasil diagnosis."
                        st.metric(label="Total Sel Terdeteksi", value="0 Bakteri", delta=kategori_teks, delta_color="normal")
                        st.info(f"**Interpretasi:** {interpretasi_teks}")
                    
                    elif 1 <= jumlah_bakteri <= 9:
                        kategori_teks = "Positif Lemah (Scanty)"
                        interpretasi_teks = "Ditemukan sejumlah kecil bakteri. Pasien terindikasi positif TB tingkat awal. Sangat disarankan untuk melakukan pengujian ulang sampel atau konfirmasi dengan Tes Cepat Molekuler (TCM)."
                        st.metric(label="Total Sel Terdeteksi", value=f"{jumlah_bakteri} Bakteri", delta=kategori_teks, delta_color="inverse")
                        st.warning(f"**Interpretasi (Scanty):** {interpretasi_teks}")
                    
                    elif 10 <= jumlah_bakteri <= 99:
                        kategori_teks = "Positif Aktif (+1)"
                        interpretasi_teks = "Terindikasi infeksi aktif tingkat sedang. Konsentrasi bakteri cukup tinggi pada dahak. Segera rujuk pasien ke fasilitas layanan kesehatan untuk memulai pengobatan OAT (Obat Anti Tuberkulosis)."
                        st.metric(label="Total Sel Terdeteksi", value=f"{jumlah_bakteri} Bakteri", delta=kategori_teks, delta_color="inverse")
                        st.error(f"**Interpretasi (+1):** {interpretasi_teks}")
                    
                    else:
                        kategori_teks = "Positif Tinggi (+2 / +3)"
                        interpretasi_teks = "Terindikasi infeksi aktif tingkat parah. Beban bakteri sangat tinggi dan pasien berisiko tinggi menularkan penyakit. Ambil tindakan isolasi dan intervensi medis darurat segera."
                        st.metric(label="Total Sel Terdeteksi", value=f"{jumlah_bakteri} Bakteri", delta=kategori_teks, delta_color="inverse")
                        st.error(f"**Interpretasi (+2 / +3):** {interpretasi_teks}")
                    
                    st.markdown("---")
                    
                    # Konversi Gambar ke Biner (Untuk didownload & ditaruh ke PDF)
                    img_pil = Image.fromarray(res_plotted[..., ::-1]) 
                    buf = io.BytesIO()
                    img_pil.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    # ========================================================
                    # PROSES PEMBUATAN DOKUMEN PDF
                    # ========================================================
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Kop Surat
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "Laporan Analisis Skrining SputumAI", ln=True, align="C")
                    
                    # Tanggal Otomatis
                    tanggal_sekarang = datetime.datetime.now().strftime('%d %B %Y - %H:%M')
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 10, f"Tanggal Cetak Dokumen: {tanggal_sekarang}", ln=True, align="C")
                    pdf.line(10, 30, 200, 30) # Garis pembatas kop surat
                    pdf.ln(10)
                    
                    # Menyisipkan Gambar Hasil AI ke PDF menggunakan Temporary File
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(byte_im)
                        tmp_path = tmp.name
                    pdf.image(tmp_path, x=20, w=170)
                    pdf.ln(5)
                    
                    # Menyisipkan Data Teks ke PDF
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, "Ringkasan Hasil Klinis", ln=True)
                    
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
                    
                    # Catatan Kaki (Disclaimer)
                    pdf.set_font("Arial", "I", 10)
                    pdf.set_text_color(100, 100, 100)
                    pdf.multi_cell(0, 5, "Peringatan (Disclaimer): Dokumen laporan ini dihasilkan secara otomatis oleh sistem Kecerdasan Buatan (AI) SputumAI untuk keperluan skrining awal. Dokumen ini BUKAN merupakan diagnosis final dari tenaga medis. Harap konsultasikan ke dokter terkait untuk tindakan lanjutan.")
                    
                    # Ekspor PDF ke dalam bentuk Byte
                    pdf_output = pdf.output(dest='S')
                    if type(pdf_output) == str:
                        pdf_bytes = pdf_output.encode('latin-1')
                    else:
                        pdf_bytes = bytes(pdf_output)

                    # ========================================================
                    # TAMPILAN 2 TOMBOL DOWNLOAD BERDAMPINGAN
                    # ========================================================
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        st.download_button(
                            label="🖼️ Unduh Gambar AI",
                            data=byte_im,
                            file_name="Hasil_Citra_SputumAI.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                    
                    with col_btn2:
                        st.download_button(
                            label="📄 Unduh Laporan PDF",
                            data=pdf_bytes,
                            file_name="Rekam_Medis_SputumAI.pdf",
                            mime="application/pdf",
                            type="primary", # Tombol diberi warna khusus
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
