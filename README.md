## 📊 Dataset & Data Preparation

Proyek ini mendemonstrasikan bagaimana Kecerdasan Buatan (AI) berbasis **Reinforcement Learning** dapat digunakan untuk mengoptimalkan strategi retensi pelanggan di industri Telekomunikasi. 
Proyek ini menggunakan dataset churn pelanggan telekomunikasi. Data tersebut diproses melalui *pipeline* berikut untuk memastikan AI belajar dari data yang bersih dan relevan:
Daripada menebak-nebak atau menyebarkan promo secara acak (yang membuang anggaran), AI ini belajar secara mandiri untuk mengalokasikan promo yang tepat (Diskon, Kuota, atau Panggilan CS) hanya kepada pelanggan yang benar-benar berisiko *churn*.

* **`Telco_Customer_Churn.csv`**: Dataset mentah (*raw data*) yang berisi informasi demografi, layanan, dan status churn pelanggan.
* **`clean_telco_data.csv`**: Hasil pembersihan data (menangani *missing values*, normalisasi fitur, dan *encoding* variabel kategorikal).
* **`train_telco.csv`**: Dataset yang digunakan khusus untuk melatih agen Reinforcement Learning.
* **`test_telco.csv`**: Dataset khusus untuk evaluasi performa model dan *stress testing* di lingkungan yang tidak terlihat (*unseen data*).

*Semua proses pembersihan data di atas didokumentasikan di dalam notebook `data_prep.ipynb`.*

## 📂 Struktur Proyek
* `telco_env.py`: Mesin simulator (Environment) tempat AI berlatih.
* `train_agent.py`: Skrip untuk melatih otak AI dari nol.
* `evaluate.py`: Skrip untuk menguji AI dan melihat grafik keuntungan bisnisnya.
* `ppo_telco_churn_model.zip`: Otak AI yang sudah pintar (hasil latihan).

---
## 📂 Ringkasan Hasil & Analisis Visual

Untuk membuktikan efektivitas sistem ini, kami telah melakukan beberapa pengujian berbasis data. Berikut adalah interpretasi dari hasil eksperimen:

### 1. Perbandingan RL vs Supervised Learning (`rl_vs_sl_comparison.png`)
* **Apa ini?**: Grafik yang membandingkan performa model *Random Forest* (Prediksi statis) dengan model *Reinforcement Learning* (Optimasi preskriptif).
* **Interpretasi**: Jika model *Random Forest* memberikan akurasi prediksi yang tinggi, model RL memberikan **Profitabilitas yang lebih tinggi**. Gambar ini membuktikan bahwa strategi retensi kita lebih efisien karena tidak membuang anggaran promo pada pelanggan yang memang tidak akan *churn*.

### 2. Uji Ketahanan Krisis (`stress_test_result.png`)
* **Apa ini?**: Simulasi bagaimana model bertahan saat biaya operasional perusahaan naik 2x lipat (inflasi/krisis ekonomi).
* **Interpretasi**: Grafik ini menunjukkan bahwa AI kita bersifat adaptif. Saat mode krisis aktif, AI secara otomatis beralih menjadi lebih selektif dalam memberikan promo (hanya memberikan promo pada pelanggan VIP), menjaga margin keuntungan perusahaan tetap positif meskipun dalam kondisi pasar yang sulit.

### 3. Evaluasi Dampak Bisnis (`business_impact_evaluation.png`)
* **Apa ini?**: Proyeksi penghematan biaya operasional (*Cost Saving*) dan kenaikan *Customer Lifetime Value* (CLV).
* **Interpretasi**: Grafik ini adalah "Bahasa Manajemen". Ini menunjukkan berapa banyak rupiah yang berhasil dihemat perusahaan dengan menggunakan AI Co-Pilot dibandingkan dengan metode manual/acak. Semakin tinggi grafiknya, semakin besar efisiensi departemen *Customer Service* kita.

## 🚀 Panduan Quick Run (Hanya Butuh VS Code)

Instruksi ini dirancang agar Anda bisa langsung menjalankan simulasi ini di komputer Anda menggunakan Visual Studio Code (VS Code).

### Langkah 1: Persiapan Folder
1. Unduh (Download) seluruh file proyek ini.
2. Buka aplikasi **VS Code**.
3. Pilih menu **File > Open Folder...** dan buka folder proyek yang baru saja Anda unduh.

### Langkah 2: Membuka Terminal & Membuat Lingkungan Virtual
Lingkungan virtual memastikan pustaka proyek ini tidak mengganggu sistem komputer Anda.
1. Di VS Code, buka terminal dengan menekan tombol **`Ctrl` + `\``** (atau klik menu **Terminal > New Terminal** di bagian atas).
2. Ketik perintah ini untuk membuat lingkungan virtual bernama "venv":
   ```bash
   python -m venv venv
   ```

3. Aktifkan lingkungan virtual tersebut:

* **Pengguna Windows:**

```Bash
.\venv\Scripts\activate
```

* **Pengguna Mac/Linux:**

```Bash
source venv/bin/activate
```

*(Tanda keberhasilan: Akan muncul tulisan `(venv)` di sebelah kiri input terminal Anda).*

### Langkah 3: Menginstal Persyaratan
Dengan lingkungan virtual yang sudah aktif, instal semua pustaka yang dibutuhkan dengan satu perintah ini:

```Bash
pip install -r requirements.txt
```
(Tunggu sekitar 1-2 menit hingga proses unduhan selesai).

### Langkah 4: Menjalankan Evaluasi Bisnis (Melihat Hasil)
Untuk melihat seberapa cerdas AI ini menghemat uang perusahaan dibandingkan penyebaran promo acak, jalankan:

```Bash
python evaluate.py
```
> **Catatan:** Skrip ini akan secara otomatis memuat data `test_telco.csv`, mengevaluasi 1.400+ pelanggan...

### Langkah 5: (Opsional) Melatih Ulang AI
Jika Anda penasaran bagaimana AI ini belajar dari nol (trial and error), Anda bisa menjalankan proses pelatihannya:

```Bash
python train_agent.py
```
Proses ini akan menjalankan 100.000 iterasi latihan dan akan menimpa/memperbarui file `ppo_telco_churn_model.zip` dengan otak AI yang baru Anda latih.

### Langkah 6:(Opsional) Luncurkan Dashboard AI
Ini adalah cara tercepat untuk melihat AI Anda bekerja secara interaktif:

```Bash
streamlit run app.py
```

### Langkah 7:(Opsional) Monitoring & Advanced Evaluation
**Cara melihat dashboard training:**
```bash
python -m tensorboard.main --logdir ./ppo_telco_tensorboard/
```
### Langkah 8:(Opsional) Stress Test 
jika biaya operasional naik atau perilaku pasar berubah tiba-tiba? Skrip stress_test.py mensimulasikan kondisi krisis untuk memastikan model Anda tetap memberikan rekomendasi yang masuk akal secara finansial.

```bash
python stress_test.py
```


