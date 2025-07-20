# 🏁 Tugas Akhir (TA) - Final Project

**Nama Mahasiswa**: Al-Ferro Yudisthira Putra <br>
**NRP**: 5025211176 <br>
**Judul TA**: KLASIFIKASI TOKOH PROTAGONIS DAN ANTAGONIS PADA CERITA RAKYAT NUSANTARA MENGGUNAKAN STRUKTUR BAHASA  <br>
**Dosen Pembimbing**: Dini Adni Navastara, S.Kom., M.Sc <br>
**Dosen Ko-pembimbing**: Prof. Dr. Diana Purwitasari, S.Kom., M.Sc <br>

---

## 📺 Demo Aplikasi  
Embed video demo di bawah ini (ganti `VIDEO_ID` dengan ID video YouTube Anda):  

[![Demo Aplikasi](https://i.ytimg.com/vi/zIfRMTxRaIs/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)  
*Klik gambar di atas untuk menonton demo*

---

*Konten selanjutnya hanya merupakan contoh awalan yang baik. Anda dapat berimprovisasi bila diperlukan.*

## 🛠 Panduan Instalasi & Menjalankan Software  

### Prasyarat  
| Library | Keterangan |
|--------|------------|
| `gensim` | Memuat word embedding (`KeyedVectors`) |
| `pandas` | Manipulasi dan analisis data tabular |
| `nltk` | Tokenisasi, stopwords, WordNet |
| `stanza` | NLP pipeline Bahasa Indonesia (POS, NER, dll.) |
| `sklearn` (`scikit-learn`) | Evaluasi model, ekstraksi fitur TF-IDF, encoding label |
| `matplotlib` | Visualisasi data |
| `Sastrawi` | Stemming Bahasa Indonesia |
| `re` (built-in) | Ekspresi reguler untuk cleaning teks |
| `ast` (built-in) | Parsing literal Python (misalnya string → list/dict) |
| `collections.Counter` (built-in) | Analisis frekuensi token |


# 📚 NLP Pipeline untuk Identifikasi Tokoh dalam Cerita Rakyat

Proyek ini merupakan pipeline Natural Language Processing (NLP) untuk Bahasa Indonesia yang digunakan dalam penelitian tugas akhir bertema identifikasi tokoh dalam cerita rakyat. Pipeline mencakup proses preprocessing teks, ekstraksi fitur, embedding, evaluasi model klasifikasi/NER, hingga visualisasi.

---

## 🛠️ Langkah-langkah

### 1. Clone Repository

```bash
git clone https://github.com/Informatics-ITS/TA.git
cd TA
```

### 2. Sesuaikan Path Dataset

Pastikan file dataset yang dipanggil pada file code sudah sesuai dengan path data.


### 3. Jalankan Script Berdasarkan Urutan Step

Pastikan environment sudah terinstall semua dan jalankan code berdasarkan STEP yang tertulis pada nama file


---

## 📚 Dokumentasi Tambahan

- [![Dokumentasi API]](docs/api.md)
- [![Diagram Arsitektur]](docs/architecture.png)
- [![Struktur Basis Data]](docs/database_schema.sql)

---

## ✅ Validasi

Pastikan proyek memenuhi kriteria berikut sebelum submit:
- Source code dapat di-build/run tanpa error
- Video demo jelas menampilkan fitur utama
- README lengkap dan terupdate
- Tidak ada data sensitif (password, API key) yang ter-expose

---

## ⁉️ Pertanyaan?

Hubungi:
- Penulis: [email@mahasiswa]
- Pembimbing Utama: [email@pembimbing]
