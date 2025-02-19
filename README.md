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
1. Membangun Model Rekomendasi dengan dua pendekatan yaitu:
- Content-Based Filtering: Menggunakan genre film dan fitur terkait untuk merekomendasikan film berdasarkan kesamaan konten dengan film yang sudah ditonton oleh pengguna.
- Collaborative Filtering: Membangun model berbasis jaringan saraf yang menggunakan embedding layer untuk menyarankan film yang belum pernah ditonton pengguna berdasarkan pola rating yang telah diberikan pengguna.
2. Menggunakan metrik evaluasi Precision untuk model dengan pendekatan Content-Based Filtering dan RMSE (Root Mean Squared Error) untuk model Collaborative Filtering guna mengukur kualitas rekomendasi yang diberikan oleh masing-masing model.


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

#### Kondisi Data
* Semua variabel berbentuk numerik dengan tipe data float64 (Open, High, Low, Close, Adj Close) dan int64 (Volume).
* Missing Value: Tidak ditemukan nilai yang hilang karena semua kolom memiliki 1692 non-null count.
* Duplikasi: Tidak ada indikasi duplikasi berdasarkan informasi yang diberikan.

![image](https://github.com/user-attachments/assets/bb780d32-3876-46cc-b56f-8a17960be6e5)

#### Statistika Deskriptif
Berdasarkan statistik deskriptif, terdapat kemungkinan outlier pada kolom Volume, di mana nilai maksimum (37,163,900) jauh lebih besar dibandingkan kuartil ketiga (5,662,100). Oleh karena itu, pada tahap selanjutnya akan dilakukan pengecekan outliers
  ![image](https://github.com/user-attachments/assets/d6607acf-9244-46b5-aaf8-7a10258764f3)



- EDA
![image](https://github.com/user-attachments/assets/db99222d-93d3-42e0-81fe-48bb35816704)

- EDA

![image](https://github.com/user-attachments/assets/7e1164c2-cd30-484e-85eb-55e57f30fe37)

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
Pada tahap ini dilakukan evaluasi terhadap hasil rekomendasi yang dihasilkan oleh model.
Hal ini bertujuan untuk menillai seberapa akurat model yang telah 
dibuat dalam memberikan rekomendasi. Untuk sistem rekomendasi dengan pendekatan Content Based Filtering menggunakan metrik evaluasi precision:
1. Precision

Precision adalah metrik evaluasi yang digunakan untuk mengukur seberapa relevan film yang direkomendasikan dibandingkan dengan jumlah total film yang direkomendasikan. Precision dihitung sebagai rasio antara jumlah film yang relevan dengan jumlah total film yang diberikan dalam daftar rekomendasi. Precision bernilai 1.0 jika semua film yang direkomendasikan memiliki minimal satu genre yang sama dengan film target pengguna, sehingga seluruh rekomendasi dianggap relevan. Precision yang lebih tinggi menunjukkan bahwa sistem rekomendasi lebih akurat dalam memberikan saran yang sesuai dengan preferensi pengguna.

Rumus Precision dalam sistem rekomendasi film adalah sebagai berikut:

$$
Precision = \frac{|Relevant_Movies \cap Recommended_Movies|}{|Recommended_Movies|}
$$

Di mana:

- **Relevant_Movies** = Daftar film yang benar-benar relevan dengan preferensi pengguna.
- **Recommended_Movies** = Daftar film yang direkomendasikan oleh sistem.


Analisis Hasil Evaluasi Rekomendasi Content Based Filtering:

- Hasil rekomendasi untuk film berjudul Dish, The (2001)
  
  ![image](https://github.com/user-attachments/assets/f3a4ff39-9398-4601-a6b2-1602832db9ab)
  
  ![image](https://github.com/user-attachments/assets/12c7036f-cf7f-49d6-90bf-12c8ba5ef33f)
  
Dalam contoh ini, precision bernilai 1.0, yang berarti semua 10 film rekomendasi memiliki minimal satu genre yang sama dengan film "Dish, The (2001)" yaitu genre Comedy, sehingga hasilnya sempurna.


Sedangkan untuk sistem rekomendasi dengan pendekatan Collaborative Filtering menggunakan metrik evaluasi Root Mean Square Error (RMSE):
1. Root Mean Square Error (RMSE)

Root Mean Square Error (MSE) merupakan salah satu metrik evaluasi yang umum 
digunakan untuk mengukur seberapa baik model memprediksi nilai tertentu. RMSE didapatkan 
dengan cara mengukur hasil akar dari rata-rata perbedaan kuadrat antara nilai aktual (y) dan 
nilai hasil prediksi (y). Rumus RMSE dapat dinyatakan sebagai berikut:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - f_i)^2}
$$


Di mana

n = Jumlah data

xi= Nilai prediksi pada periode ke-i 

fi = Nilai actual indeks pada periode ke-i


Analisis Hasil Evaluasi Sistem Rekomendasi Collaborative Filtering :

![image](https://github.com/user-attachments/assets/4f229875-a001-4c90-bbe5-296deeda00fa)

- Berdasarkan plot grafik tersebut, proses training model cukup smooth dan model konvergen pada epoch sekitar 8.

![image](https://github.com/user-attachments/assets/fa90674f-dd75-4c11-8ffe-43600ee85951)

- Nilai error RMSE akhir pada data training sebesar 0.2023 dan error RMSE pada data validasi sebesar 0.2037. Nilai ini menunjukkan bahwa model memiliki generalisasi yang baik karena perbedaan error antara training dan validasi sangat kecil. 
- Nilai RMSE yang rendah ini mengindikasikan bahwa rata-rata selisih antara prediksi model dan nilai rating asli cukup kecil, sehingga model dapat memberikan rekomendasi yang cukup akurat. Selain itu, tidak terdapat indikasi overfitting karena kurva training dan validasi berjalan sejajar tanpa ada kesenjangan yang signifikan.


### Dampak terhadap Business Understanding

Sistem rekomendasi film yang dikembangkan memiliki dampak signifikan terhadap pemahaman bisnis, terutama dalam hal meningkatkan pengalaman pengguna di platform streaming. Berikut adalah beberapa aspek utama dampaknya:

1. Menjawab Problem Statement dan Mencapai Goals

- Sistem rekomendasi film dengan pendekatan Content Based Filtering berhasil memberikan Top-N rekomendasi film berdasarkan genre favorit pengguna.

- Sistem rekomendasi film dengan pendekatan Collaborative Filtering berhasil memberikan rekomendasi film yang belum pernah ditonton oleh pengguna berdasarkan rating yang telah diberikan.
  
- Berdasarkan evaluasi menggunakan metrik Precision pada sistem rekomendasi Content Based Filterting dan RMSE pada sistem rekomendasi Collaborative Filtering menunjukkan bahwa kedua model sistem rekomendasi memiliki kinerja yang baik dalam memberikan rekomendasi yang relevan dan akurat, serta memenuhi tujuan untuk memberikan rekomendasi film yang lebih personal dan sesuai dengan preferensi pengguna.


2. Dampak dari Solusi Statement

- Sistem Rekomendasi dengan pendeketan Content Based Filtering dan Collaborative Filtering keduanya berhasil memberikan hasil rekomendasi yang akurat dan relevan yang dibuktikan dengan nilai Evaluasi menggunakan metrik Precision untuk Content-Based Filtering dan RMSE untuk Collaborative Filtering menunjukkan nilai yang sangat baik.

- Dengan adanya sistem rekomendasi film ini, dapat dijadikan pertimbangan bagi pihak platform streaming film dalam meningkatkan pengalaman pengguna dengan cara yang lebih personal dan menyenangkan sehingga mendorong pengguna untuk lebih lama bertahan di platform dan memperbesar peluang peningkatan engagement.

## Kesimpulan
Model dengan pendekatan Content-Based Filtering dan Collaborative Filtering mampu memberikan rekomendasi yang relevan dengan baik kepada pengguna. Content-Based Filtering lebih fokus pada kesamaan genre film yang sudah ditonton pengguna, sementara Collaborative Filtering berbasis jaringan saraf, memanfaatkan pola rating dari pengguna lain untuk menyarankan film yang belum pernah ditonton. Kedua pendekatan ini memberikan nilai tambah bagi platform streaming sehingga diharapkan dapat meningkatkan pengalaman pengguna dengan rekomendasi yang lebih personal dan sesuai selera.


## Refrensi


