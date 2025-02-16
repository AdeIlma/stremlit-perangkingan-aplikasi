# Google Play Reviews Scraper, Sentiment Analysis, and SAW Ranking
## Deskripsi
Repository ini berisi kode untuk melakukan scraping ulasan aplikasi dari Google Play Store, melakukan preprocessing teks, menerjemahkan teks, melakukan analisis sentimen menggunakan VADER, serta melakukan perankingan aplikasi berdasarkan sentimen menggunakan metode Simple Additive Weighting (SAW).
## Metode SAW (Simple Additive Weighting)
Metode SAW digunakan untuk menentukan peringkat aplikasi berdasarkan analisis sentimen ulasan pengguna. Tahapan yang dilakukan dalam metode SAW:
#### 1. Menentukan Kriteria
   - Positif Sentiment (Benefit) → Semakin tinggi, semakin baik.
   - Negatif Sentiment (Cost) → Semakin rendah, semakin baik.
   - Netral Sentiment → Tidak langsung mempengaruhi peringkat, tetapi digunakan dalam perhitungan.
#### 2. Normalisasi 
Normalisasi dilakukan menggunakan rumus:
1. Untuk benefit (semakin tinggi semakin baik):
   ![image](https://github.com/user-attachments/assets/e3733323-054a-4d59-80c4-fbc7a7cf49a7)

2. Untuk cost (semakin rendah semakin baik):
   ![image](https://github.com/user-attachments/assets/4c07e4dd-7f64-4b3f-84fb-c8c6f7a676de)

3. Untuk netral, digunakan rata-rata:
   ![image](https://github.com/user-attachments/assets/79326d2c-0ffb-426d-8ccc-55388facebde)

#### 3. Menghitung Nilai Preferensi
Setelah normalisasi, setiap alternatif dihitung menggunakan persamaan:
![image](https://github.com/user-attachments/assets/156b0e48-257f-47c8-a8ca-a80221af41b6)
Di mana:
1. Vi = nilai preferensi
2. wj = bobot masing-masing kriteria
3. rij =  nilai normalisasi untuk setiap kriteria

#### 4. Menentukan Peringkat
Aplikasi dengan nilai Vi tertinggi dianggap sebagai aplikasi terbaik berdasarkan analisis sentimen pengguna.

                                
