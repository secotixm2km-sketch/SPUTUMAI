from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
from PIL import Image
import io

app = Flask(__name__)

# Load model YOLO
model_path = "best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
else:
    model = None

# Route utama untuk menampilkan halaman HTML
@app.route('/')
def home():
    return render_template('index.html')

# Route API untuk menerima gambar dan memproses deteksi AI
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model best.pt tidak ditemukan."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File kosong."}), 400

    try:
        # Baca gambar yang dikirim dari HTML
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Jalankan pemindaian AI
        results = model.predict(source=image, conf=0.1, imgsz=640, verbose=False)[0]
        boxes = results.boxes
        count = len(boxes) if boxes is not None else 0
        
        confidences = [float(c) for c in boxes.conf.tolist()] if boxes is not None and count > 0 else []
        avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else 0.0

        # Tentukan status
        if count == 0:
            status = "NEGATIF (Tidak Ditemukan BTA)"
        elif 1 <= count <= 9:
            status = "SCANTY (BTA Positif Rendah)"
        else:
            status = "POSITIF (BTA Ditemukan > 9)"

        # Kembalikan hasil ke HTML dalam format JSON
        return jsonify({
            "bta_count": count,
            "confidence": round(avg_conf, 1),
            "status": status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Jalankan server
    app.run(debug=True, port=8000)
