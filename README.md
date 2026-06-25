# Web Streamlit — Klasifikasi Penyakit Kambing

Aplikasi ini memakai model `best_efficientnetb0_kambing.keras` hasil training dari notebook Colab.

## Struktur folder yang dibutuhkan

```text
web_streamlit_kambing/
├── app.py
├── requirements.txt
└── model/
    ├── best_efficientnetb0_kambing.keras
    ├── class_names.json
    └── threshold.json
```

File model diambil dari hasil notebook:

```text
/content/drive/MyDrive/skripsi_kambing/model/best_efficientnetb0_kambing.keras
/content/drive/MyDrive/skripsi_kambing/model/class_names.json
/content/drive/MyDrive/skripsi_kambing/model/threshold.json
```

Salin tiga file tersebut ke folder `model/`.

## Cara menjalankan di laptop / VS Code

```bash
cd web_streamlit_kambing
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di browser, biasanya pada:

```text
http://localhost:8501
```

## Cara menjalankan dengan model di folder lain

Jika model tidak diletakkan di folder `model/`, jalankan dengan environment variable:

```bash
MODEL_DIR="D:/Project/skripsi_kambing/model" streamlit run app.py
```

Untuk Windows PowerShell:

```powershell
$env:MODEL_DIR="D:\Project\skripsi_kambing\model"
streamlit run app.py
```

## Catatan penggunaan

Model dilatih pada tiga kelas penyakit: `scabies`, `orf`, dan `pink_eye`.

Folder `sehat` pada data uji eksternal tidak menjadi kelas training. Karena itu, aplikasi menggunakan confidence threshold. Jika confidence rendah, aplikasi menampilkan peringatan bahwa hasil belum meyakinkan atau kemungkinan citra berada di luar tiga kelas utama.