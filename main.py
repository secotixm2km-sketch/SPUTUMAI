from ultralytics import YOLO

# 1. Load model AI kamu
model = YOLO('best.pt')

# 2. Deteksi gambar sputum test kamu
results = model('sputum_test_0001.jpg')

# 3. Tampilkan hasil dan simpan
results[0].show()
results[0].save(filename='hasil_deteksi.jpg')

print("Selesai! Hasil deteksi sudah disimpan.")