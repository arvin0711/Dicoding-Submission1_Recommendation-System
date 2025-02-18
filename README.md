# Laporan Proyek Machine Learning - Arvin Azmi Sava
## Domain Proyek
Proyek ini bertujuan untuk membangun sebuah sistem rekomendasi film menggunakan teknik content-based filtering dan collaborative filtering. Sistem rekomendasi ini akan memberikan saran film berdasarkan preferensi pengguna yang ada, menggunakan dataset yang mencakup data rating film dan metadata terkait film tersebut.

## Business Understanding
Sistem rekomendasi film memiliki potensi untuk memberikan manfaat signifikan bagi platform streaming. Dengan sistem ini, platform dapat meningkatkan engagement pengguna dengan menyarankan film yang sesuai dengan selera mereka, serta memberikan rekomendasi yang lebih personal dan relevan.
### Problem Statements
1. Bagaimana cara membangun sistem rekomendasi yang merekomnedasikan pengguna berdasarkan genre film?
2. Bagaimana cara membuat sistem rekomendasi untuk merekomendasikan film yang belum ditonton oleh pengguna berdasarkan data rating mereka?
3. Bagaimana cara mengukur performa dari model rekomendasi yang telah dibangun?
### Goals
1. Membangun sistem rekomendasi yang dapat memberikan Top-N rekomendasi film berdasarkan genre favorit pengguna, sehingga pengguna dapat dengan mudah menemukan film dengan genre yang mereka sukai.
2. Mengembangkan model rekomendasi yang dapat menyarankan film yang belum ditonton oleh pengguna berdasarkan rating yang telah diberikan untuk memberikan pengalaman menonton yang lebih personal.
3. Mengevaluasi performa model rekomendasi dengan menggunakan berbagai metrik evaluasi untuk memastikan rekomendasi yang akurat dan bermanfaat bagi pengguna.


### Solution Statements
1. Menggunakan 
2. 


## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan dataset film yang digunakan untuk membangun sistem rekomendasi berdasarkan preferensi pengguna. Dataset ini terdiri dari dua file yaitu movies.csv dan ratings.csv. Pada file movies.csv terdiri dari 3 kolom dengan 9742 baris data. Sedangkan, pada file rating.csv terdiri dari 4 kolom dengan 100836 baris data. Dataset ini diperoleh dari Kaggle melalui tautan berikut: (https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system)

### Variabel dalam file movies.csv:
- movieId - ID unik untuk setiap film.
- title - Judul film.
- genres - Genre film (misalnya, aksi, drama, komedi, dll).

### Variabel dalam file ratings.csv:
- userId - ID unik untuk setiap pengguna yang memberikan rating.
- movieId - ID film yang diberi rating oleh pengguna.
- rating - Rating yang diberikan oleh pengguna
- timestamp - Waktu ketika rating diberikan.
  
### Exploratory Data Analysis (EDA)
#### Plot harga saham historis untuk melihat tren.

![image](https://github.com/user-attachments/assets/71545ac0-de14-46c2-932b-1aa6f8c5b51c) ![image](https://github.com/user-attachments/assets/5652cc40-36da-40ec-9c3b-9b171efaddc2)

#### Kondisi Data
* Semua variabel berbentuk numerik dengan tipe data float64 (Open, High, Low, Close, Adj Close) dan int64 (Volume).
* Missing Value: Tidak ditemukan nilai yang hilang karena semua kolom memiliki 1692 non-null count.
* Duplikasi: Tidak ada indikasi duplikasi berdasarkan informasi yang diberikan.

![image](https://github.com/user-attachments/assets/bb780d32-3876-46cc-b56f-8a17960be6e5)

#### Statistika Deskriptif
Berdasarkan statistik deskriptif, terdapat kemungkinan outlier pada kolom Volume, di mana nilai maksimum (37,163,900) jauh lebih besar dibandingkan kuartil ketiga (5,662,100). Oleh karena itu, pada tahap selanjutnya akan dilakukan pengecekan outliers
  ![image](https://github.com/user-attachments/assets/d6607acf-9244-46b5-aaf8-7a10258764f3)


## Data Preparation
- Deteksi Outlier
  ![image](https://github.com/user-attachments/assets/be12abff-9cc9-4441-b461-94ef576e0954)

  Berdasarkan visualisasi boxplot, terdapat outlier pada variabel Volume. Oleh karena itu, akan dilakukan pengecekan outlier tersebut. Kemudian, berikutnya outlier tersebut akan dihapus / didrop karena jumlahnya tidak begitu banyak sehingga tidak berpengaruh signifikan terhadap dataset yang ada

  ![image](https://github.com/user-attachments/assets/71a5201c-7e43-48c2-b3b4-86e70eec2b5a)

  Penghapusan outliers dilakukan menggunakan metode IQR. Outlier berjumlah 80. Setelah dilakukan penghapusan baris data yang mengandung outliers, saat ini total baris data berjumlah 1612 dari sebelumnya berjumlah 1692 baris.


- Pembagian Data Latih dan Uji

Pada tahap ini dilakukan pembagian data menjadi menjadi dua bagian yaitu data latih dan data uji. Pembagian data latih dan data uji ini menggunakan  skenario 80:20. Skenario ini dirancang untuk menguji seberapa baik kemampuan model dalam memprediksi data yang belum pernah dilihat sebelumnya. Setelah pembagian data, dilakukan penentuan variabel fitur dan target. Variabel fitur terdiri dari kolom Open, High, Low, dan Volume, sedangkan variabel targetnya yaitu Close. 
- Normalisasi

Normalisasi adalah proses mengubah nilai-nilai dari suatu dataset ke dalam rentang nilai tertentu. Tujuan utama normalisasi adalah untuk menghasilkan data yang konsisten sehingga setiap variabel memiliki pangaruh yang seimbang terhadap model yang dibangun [2]. Selain itu, normalisasi juga akan mengurangi bias yang mungkin terjadi akibat perbedaan skala antar variabel. Dengan demikian, normalisasi sangat penting dalam pembuatan model karena dapat menghasilkan model yang lebih stabil dan akurat. Hasil normalisasi pada variabel fitur ditunjukkan pada gambar di bawah ini

![image](https://github.com/user-attachments/assets/a18ac943-5bc1-494e-99c6-af4881f5eddc)

Sementara itu, untuk hasil normalisasi variabel target ditunjukkan gambar di bawah ini

![image](https://github.com/user-attachments/assets/b553097d-c425-49df-8ae9-cc2d1b7a8806)


- Pembuatan Urutan Data Baru

Setelah melakukan normalisasi data, tahap berikutnya adalah pembuatan urutan data baru menjadi ukuran 3 dimensi (samples, timesteps, jumlah fitur) agar sesuai dengan input yang diperlukan oleh model biLSTM. Timesteps digunakan untuk menentukan jumlah data masa lalu yang diperhitungkan dalam memprediksi satu nilai di masa depan. Pada tahap ini timesteps yang digunakan adalah 9 sehingga setiap prediksi di masa depan mempertimbangkan 9data di masa lalu. Pada Gambar di bawah ini ditunjukkan bahwa ukuran data latih dan data uji sudah berubah menjadi ukuran 3 dimensi.

![image](https://github.com/user-attachments/assets/a50c404a-4a97-4c35-bb27-fcc4e3c3b660)

## Modeling


## Evaluation
Setelah memperoleh hasil prediksi dari model yang telah dibangun. Perlu dilakukan 
evaluasi kinerja model. Hal ini bertujuan untuk menillai seberapa akurat model yang telah 
dibuat dalam memprediksi data. Berikut ini beberapa metrik evaluasi kinerja yang digunakan 
untuk menilai seberapa baik peramalan yang dihasilkan:
1. Mean Square Error (MSE)

Mean Square Error (MSE) merupakan salah satu metrik evaluasi yang umum 
digunakan untuk mengukur seberapa baik model memprediksi nilai tertentu. MSE didapatkan 
dengan cara mengukur hasil akar dari rata-rata perbedaan kuadrat antara nilai aktual (y) dan 
nilai hasil prediksi (y). Rumus MSE dapat dinyatakan sebagai berikut:

   $$
   MSE = \frac{1}{n} \sum_{i=1}^{n} (x_i - f_i)^2
   $$

Di mana

n = Jumlah data

xi= Nilai prediksi pada periode ke-i 

fi = Nilai actual indeks pada periode ke-i


Analisis Hasil Evaluasi:

- MSE (Mean Squared Error): BiGRU memiliki nilai MSE yang lebih rendah dibandingkan BiLSTM, menunjukkan bahwa model ini menghasilkan error yang lebih kecil.


### Dampak terhadap Business Understanding

Model prediksi harga saham yang dikembangkan memiliki dampak signifikan terhadap pemahaman bisnis, terutama dalam pengambilan keputusan investasi. Berikut adalah beberapa aspek utama dampaknya:

1. Menjawab Problem Statement dan Mencapai Goals

- Model yang dikembangkan mampu memprediksi harga saham berdasarkan data historis dengan tingkat akurasi yang baik, membantu investor dalam membuat keputusan lebih tepat.

- Penggunaan BiLSTM dan BiGRU menunjukkan bahwa deep learning efektif dalam memprediksi harga saham, metrik evaluasi menunjukkan bahwa BiGRU lebih optimal dengan error yang lebih rendah dan nilai R² yang lebih tinggi
  
- Variabel fitur Open, High, dan Low memiliki pengaruh besar terhadap variabel target Close (mendekati 1) dalam prediksi harga saham. Sementara itu, variabel fitur Volume menunjukkan korelasi yang rendah dengan variabel Close (0,40) yang menunjukkan bahwa perubahan harga tidak selalu berkaitan dengan jumlah volume perdagangan.

2. Dampak dari Solusi Statement

- Penggunaan BiLSTM dan BiGRU membuktikan bahwa deep learning dapat menangkap pola harga saham dengan baik, memberikan metode yang lebih canggih dibandingkan pendekatan konvensional.

- Hasil tuning hyperparameter yang dilakukan berhasil meningkatkan kinerja model secara signifikan, menjadikan model lebih akurat dan andal.

- Model yang dikembangkan dapat menjadi acuan bagi investor dan analis dalam pengambilan keputusan bisnis terkait investasi saham, sehingga dapat meminimalkan risiko dan memaksimalkan keuntungan.
## Kesimpulan
Berdasarkan hasil evaluasi, model algoritma BiGRU dipilih sebagai model terbaik untuk memprediksi harga saham. Hal ini didasarkan pada metrik evaluasi yang menunjukkan bahwa BiGRU memiliki error yang lebih rendah (MSE, MAE, dan MAPE) serta nilai R² yang lebih tinggi dibandingkan BiLSTM. Selain itu, BiGRU lebih efisien dalam proses pelatihan dibandingkan BiLSTM, sehingga lebih optimal untuk digunakan dalam peramalan harga saham.


## Refrensi


