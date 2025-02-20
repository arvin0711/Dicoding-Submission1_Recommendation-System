# Laporan Proyek Machine Learning - Arvin Azmi Sava
## Domain Proyek
Proyek ini bertujuan untuk membangun sebuah sistem rekomendasi film menggunakan teknik content-based filtering dan collaborative filtering. Sistem rekomendasi ini akan memberikan saran film berdasarkan preferensi pengguna yang ada, menggunakan dataset yang mencakup data rating film dan metadata terkait film tersebut.

## Business Understanding
Sistem rekomendasi film memiliki potensi untuk memberikan manfaat signifikan bagi platform streaming. Dengan sistem ini, platform dapat meningkatkan engagement pengguna dengan menyarankan film yang sesuai dengan selera mereka, serta memberikan rekomendasi yang lebih personal dan relevan. Studi sebelumnya menunjukkan bahwa sistem rekomendasi dapat meningkatkan kepuasan pengguna dan waktu yang dihabiskan dalam platform dengan memberikan saran yang lebih relevan berdasarkan analisis perilaku pengguna [3]. Selain itu, Implementasi sistem rekomendasi yang efisien juga dapat membantu platform streaming dalam memaksimalkan pendapatan dengan meningkatkan retensi pengguna [5].
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

![image](https://github.com/user-attachments/assets/799ce29e-fc8f-4462-993a-29d8be3edd8d)

### Variabel dalam file ratings.csv:
- userId - ID unik untuk setiap pengguna yang memberikan rating.
- movieId - ID film yang diberi rating oleh pengguna.
- rating - Rating yang diberikan oleh pengguna
- timestamp - Waktu ketika rating diberikan.

![image](https://github.com/user-attachments/assets/93d414b1-01ea-46e7-b578-abcd048d9bda)

## Data Preparation

#### Kondisi Data movies.csv
* variabel title dan genres memiliki tipe data object. Sementara variabel movieId bertipe data Integer.
* Missing Value: Tidak ditemukan nilai yang hilang karena semua kolom memiliki 9742 non-null count.
* Duplikasi: Tidak ada indikasi duplikasi berdasarkan informasi yang diberikan.


![image](https://github.com/user-attachments/assets/6ee3980b-2f9c-4e63-b8ce-c826da1b5661)

#### Kondisi Data ratings.csv
* Semua variabel berbentuk numerik dengan tipe data int64 (userId, movieId, timestamp) dan int64 (rating).
* Missing Value: Tidak ditemukan nilai yang hilang karena semua kolom memiliki 100836 non-null count.
* Duplikasi: Tidak ada indikasi duplikasi berdasarkan informasi yang diberikan.
  
![image](https://github.com/user-attachments/assets/d15aed6c-c5e0-4b9d-86fc-f62c7302f043)

#### Statistika Deskriptif
Berdasarkan statistik deskriptif pada dataframe ratings.csv, variabel rating memiliki nilai terkecil 0.5 dan nilai terbesar 5, sehingga rentang nilai rating yang diberikan user antara 0.5 sampai dengan 5

![image](https://github.com/user-attachments/assets/b3feb801-66b8-4d2f-b9e6-99004ea19331)




#### Visualiasi Genre Film
Berdasarkan visualisasi grafik batang di bawah. Dapat dilihat bahwa genres film terbanyak secara berurutan adalah genre Drama, Comedy, Thriller, Action, dst. Informasi ini dapat digunakan oleh perusahaan penyedia streaming film untuk merekomendasikan film dengan genre-genres tertentu yang memang paling disukai oleh pengguna.

![image](https://github.com/user-attachments/assets/db99222d-93d3-42e0-81fe-48bb35816704)

#### Visualisasi Persebaran Nilai Rating
Berdasarkan gambar persebaran nilai rating yang diberikan user, terlihat bahwa nilai terbanyak adalah nilai dengan rating 4. Kemudian, disusul dengan nilai rating 3. Sementara itu, nilai terkecil yang diberikan user adalah 0,5 dan nilai terbesar adalah 5.

![image](https://github.com/user-attachments/assets/7e1164c2-cd30-484e-85eb-55e57f30fe37)






## Model Development Content Based Filtering

# Collaborative Filterting

## Data Preparation
### Penggabungan Dataset  Movies.csv dan Ratings.csv
Dataset yang digunakan untuk model dengan pendekatan Collaborative Filtering menggunakan data hasil penggabungan dari movies.csv dan ratings.csv. Oleh karena itu, dilakukan penggabungan dua dataframe yaitu movies.csv dan ratings.csv (df2 dan df1) berdasarkan kolom movieId yang ada di kedua dataframe. Proses ini menggunakan metode penggabungan inner join, yang hanya akan menyertakan baris yang memiliki kecocokan pada kolom movieId di kedua dataframe. Hasilnya adalah dataframe baru df yang berisi informasi gabungan dari kedua dataframe tersebut

![image](https://github.com/user-attachments/assets/d572074b-b776-4315-bc7c-34465ed7b265)

### Kondisi Data
* Variabel userId, movieId, timestamp memiliki tipe data int64. Kemudian, variabel rating memiliki tipe data float64. Sedangkan, title dan genres bertipe data object
* Missing Value: Tidak ditemukan nilai yang hilang karena semua kolom memiliki 100836 non-null count.
* Duplikasi: Tidak ada indikasi duplikasi berdasarkan informasi yang diberikan.
![image](https://github.com/user-attachments/assets/2cef9fa1-c38e-4aa3-b37e-02f5b3ad0d4e)

- Pengecekan Missing Value
Tidak ditemukan missing value pada dataframe hasil penggabungan ini

![image](https://github.com/user-attachments/assets/a9819b3e-3ae2-4e97-bf14-ab03460ed573)

- Encode fitur userId dan movieId
  
Pada tahap ini dilakukan persiapan data untuk menyandikan (encode) fitur userId dan movieId ke dalam indeks integer. Kemudian, dilakukan pemetaan userId dan movieId ke dalam dataframe yang berkaitan
![image](https://github.com/user-attachments/assets/5f5a0c33-e3b4-44dc-ab52-2fb9bdbdbadf)

- Pengecek Jumlah User dan Jumlah FIlm
  
Pada tahap ini dilakukan pengecekan jumlah user dan film. Didapatkan bahwa jumlah user adalah sebanyak 610 dan jumlah film sebanyak 9724. Selain itu, dilakukan pengecekan nilai minimum rating yang diberikan user yaitu 0.5 dan nilai maksimalnya 5

![image](https://github.com/user-attachments/assets/801818dc-2cc6-4d19-805d-af03aed71026)

- Pembagian Data Latih dan Validasi

Pada tahap ini dilakukan pembagian data menjadi data training dan validasi dengan proporsi 80:20. Namun, sebelum melakukan pembagian data training dan validasi perlu dilakukan penentuan variabel fitur dan target. Variabel fiturnya sendiri ada dua yaitu user dan movie. Sedangkan variabel target berupa rating film yang telah dinormalisasi dalam rentang 0 hingga 1 menggunakan Min-Max Scaling. Tujuan utama normalisasi adalah untuk menghasilkan data yang konsisten sehingga setiap variabel memiliki pangaruh yang seimbang terhadap model yang dibangun [2].

![image](https://github.com/user-attachments/assets/fc3620ed-92b8-4561-8170-b20b0cc4754f)




## Model Development Collaborative Filtering
Model neural network yang digunakan dalam sistem rekomendasi ini terdiri dari beberapa lapisan utama. Dua lapisan pertama adalah embedding layers yang digunakan untuk merepresentasikan ID pengguna dan ID film dalam bentuk vektor berdimensi 50. Output dari embedding layers ini kemudian diratakan menggunakan lapisan Flatten. Setelah itu, hasilnya digabungkan menggunakan lapisan Concatenate untuk menghasilkan representasi gabungan pengguna dan film.

Selanjutnya, model memiliki lapisan Dense dengan 64 neuron yang berfungsi untuk menangkap hubungan kompleks antara pengguna dan film. Untuk meningkatkan stabilitas pelatihan, model juga menggunakan lapisan normalisasi BatchNormalization serta lapisan Dropout untuk mencegah overfitting. Pada lapisan terakhir, model memiliki lapisan Dense dengan satu neuron keluaran yang berfungsi untuk memprediksi skor rekomendasi film bagi pengguna.

![image](https://github.com/user-attachments/assets/d4ffd8be-2c9c-47d6-99ee-3076a628fc6d)

### Visualisasi Metrik
Berdasarkan plot grafik di bawah ini, proses training model cukup smooth dan model konvergen pada epoch sekitar 5.  Model menunjukkan kemampuan generalisasi yang baik, ditandai dengan perbedaan error yang sangat kecil antara data training dan validasi. Selain itu, tidak terdapat indikasi overfitting, karena kurva training dan validasi bergerak sejajar tanpa kesenjangan yang signifikan.

![image](https://github.com/user-attachments/assets/24c917bc-f2fc-4260-8aed-c2196e6452ca)

### Mendapatkan Rekomendasi Film
Untuk memberikan rekomendasi film, sistem pertama-tama menerima ID pengguna dan memastikan bahwa ID tersebut sudah sesuai. Selanjutnya, sistem mengumpulkan daftar film yang sudah pernah ditonton oleh pengguna. Setelah itu, daftar film yang akan diprediksi dibuat dengan mengambil semua film yang tersedia, lalu menghapus film yang sudah ditonton agar hanya film baru yang direkomendasikan. ID film yang akan diprediksi kemudian diproses agar sesuai dengan format yang dibutuhkan. Sistem menggabungkan ID pengguna dengan ID film yang akan diprediksi dan menggunakannya untuk memperkirakan rating setiap film. Terakhir, sistem memilih 10 film dengan perkiraan rating tertinggi untuk direkomendasikan. Dengan pendekatan ini, rekomendasi yang diberikan lebih relevan karena didasarkan pada preferensi pengguna dan film yang belum pernah ditonton.
![image](https://github.com/user-attachments/assets/1a3b8b98-c975-4001-b887-f1d556db2270)

Model berhasil memberikan rekomendasi film kepada user. Sebagai contoh, hasil di atas adalah rekomendasi untuk user dengan id 113. Dari output tersebut,dapat dibandingkan antara Film with high ratings from user dan Top 10 film recommendation untuk user. Dapat dilihat bahwa beberapa film rekomendasi memiliki genres yang sesuai dengan rating user.



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

![image](https://github.com/user-attachments/assets/6f812783-45a6-463f-8601-1f4fcbd9bc8e)

- Nilai error RMSE akhir pada data training sebesar 0.1973 dan error RMSE pada data validasi sebesar 0.2005. Nilai ini menunjukkan bahwa model memiliki generalisasi yang baik karena perbedaan error antara training dan validasi sangat kecil. 
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
[3] C. P. Wijaya, "Systematic Literature Review pada Sistem Rekomendasi Film untuk Layanan Streaming," Jurnal Sistem Informasi, vol. 10, no. 1, pp. 30-40, 2024.

[5] F. Putra, "Sistem Rekomendasi untuk Maksimalisasi Industri Film dengan Metode Demographic Filtering dan Content-Based Filtering," Jurnal Teknologi dan Manajemen, vol. 8, no. 2, pp. 55-65, 2023.

