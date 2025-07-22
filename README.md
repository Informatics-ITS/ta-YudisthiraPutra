# 🏁 Tugas Akhir (TA) - Final Project

**Nama Mahasiswa**: Al-Ferro Yudisthira Putra <br>
**NRP**: 5025211176 <br>
**Judul TA**: KLASIFIKASI TOKOH PROTAGONIS DAN ANTAGONIS PADA CERITA RAKYAT NUSANTARA MENGGUNAKAN STRUKTUR BAHASA <br>
**Dosen Pembimbing**: Dini Adni Navastara, S.Kom., M.Sc <br>
**Dosen Ko-pembimbing**: Prof. Dr. Diana Purwitasari, S.Kom., M.Sc <br>

---

## 📺 Demo Aplikasi


[![Demo Aplikasi](https://img.youtube.com/vi/jWZ0UluRa9k/0.jpg)](https://www.youtube.com/watch?v=jWZ0UluRa9k)
 
_Klik gambar di atas untuk menonton demo_

---


## 🛠 Panduan Instalasi & Menjalankan Software

### Prasyarat

| Library                          | Keterangan                                             |
| -------------------------------- | ------------------------------------------------------ |
| `gensim`                         | Memuat word embedding (`KeyedVectors`)                 |
| `pandas`                         | Manipulasi dan analisis data tabular                   |
| `nltk`                           | Tokenisasi, stopwords, WordNet                         |
| `stanza`                         | NLP pipeline Bahasa Indonesia (POS, NER, dll.)         |
| `sklearn` (`scikit-learn`)       | Evaluasi model, ekstraksi fitur TF-IDF, encoding label |
| `matplotlib`                     | Visualisasi data                                       |
| `Sastrawi`                       | Stemming Bahasa Indonesia                              |
| `re` (built-in)                  | Ekspresi reguler untuk cleaning teks                   |
| `ast` (built-in)                 | Parsing literal Python (misalnya string → list/dict)   |
| `collections.Counter` (built-in) | Analisis frekuensi token                               |

# 📚 NLP Pipeline untuk Identifikasi Tokoh dalam Cerita Rakyat

Proyek ini merupakan pipeline Natural Language Processing (NLP) untuk Bahasa Indonesia yang digunakan dalam penelitian tugas akhir bertema identifikasi tokoh dalam cerita rakyat. Pipeline mencakup proses preprocessing teks, ekstraksi fitur, embedding, evaluasi model klasifikasi/NER, hingga visualisasi.

---

## 🛠️ Langkah-langkah

### 1. Clone Repository

```bash
git clone git@github.com:Informatics-ITS/ta-YudisthiraPutra.git
```

### 2. Sesuaikan Path Dataset

Pastikan file dataset yang dipanggil pada file code sudah sesuai dengan path data.

### 3. Jalankan Script Berdasarkan Urutan Step

Pastikan environment sudah terinstall semua dan jalankan code berdasarkan STEP yang tertulis pada nama file

---

## 🛠️ Langkah-langkah Run Visualisasi

### 1. Clone Repository

```bash
git clone git@github.com:Informatics-ITS/ta-YudisthiraPutra.git
```

### 2. Clone Repository

```bash
pip install requirements.txt
```

### 3. Sesuaikan Path yang Digunakan

Pastikan path-path data yang digunakan telah sesuai

### 4. Jalankan Streamlit

```bash
streamlit run visualisation.py
```

### 5. Sesuaikan Path yang Digunakan

Pergi ke http://localhost:8501/

---

## ⁉️ Pertanyaan?

Hubungi:

- Penulis: [email@mahasiswa]
- Pembimbing Utama: [dini_navastara@if.its.ac.id]
