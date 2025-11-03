# k-Nearest Neighbors (kNN), Naive Bayes (NB), dan k-Fold Cross-Validation — Perhitungan Manual


## 1) k-Nearest Neighbors (kNN)

### 1.1 Intuisi

* kNN **mengingat** data latih.
* Prediksi titik baru $T$:

  1. Hitung **jarak** $d(T, x_i)$ ke semua titik latih.
  2. Urutkan jarak dari kecil ke besar.
  3. Ambil **$k$ tetangga terdekat**, lakukan **voting** mayoritas (klasifikasi) atau rata-rata (regresi).

**Parameter penting:** $k$ (mis. 3, 5), metrik jarak (umumnya **Euclidean**), dan **scaling** fitur numerik.

### 1.2 Rumus jarak Euclidean

Untuk dua vektor berdimensi $d$, $\mathbf{p}=(p_1,\dots,p_d)$ dan $\mathbf{q}=(q_1,\dots,q_d)$:

* $d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{d} (p_i - q_i)^2}$

Khusus 2D $(x,y)$:

* $d\big((x_p,y_p),(x_q,y_q)\big) = \sqrt{(x_p-x_q)^2 + (y_p-y_q)^2}$

### 1.3 Contoh hitung manual (2D, klasifikasi, $k=3$)

**Data latih:**

| Titik | $x$ | $y$ | Kelas |
| ----- | --: | --: | ----: |
| A     |   1 |   1 |     0 |
| B     |   2 |   2 |     0 |
| C     |   1 |   2 |     0 |
| D     |   5 |   5 |     1 |
| E     |   6 |   5 |     1 |
| F     |   5 |   6 |     1 |

**Titik uji:** $T=(3,3)$.

Hitung jarak Euclidean:

* $d(T,A)=\sqrt{(3-1)^2+(3-1)^2}=\sqrt{4+4}=\sqrt{8}\approx 2.828$
* $d(T,B)=\sqrt{(3-2)^2+(3-2)^2}=\sqrt{1+1}=\sqrt{2}\approx 1.414$
* $d(T,C)=\sqrt{(3-1)^2+(3-2)^2}=\sqrt{4+1}=\sqrt{5}\approx 2.236$
* $d(T,D)=\sqrt{(3-5)^2+(3-5)^2}=\sqrt{4+4}=\sqrt{8}\approx 2.828$
* $d(T,E)=\sqrt{(3-6)^2+(3-5)^2}=\sqrt{9+4}=\sqrt{13}\approx 3.606$
* $d(T,F)=\sqrt{(3-5)^2+(3-6)^2}=\sqrt{4+9}=\sqrt{13}\approx 3.606$

**Urut terdekat (kecil → besar):** B (1.414, 0), C (2.236, 0), A (2.828, 0), D (2.828, 1), E (3.606, 1), F (3.606, 1).
Ambil $k=3$ → ${B,C,A}$ → mayoritas **kelas 0** → **Prediksi: 0**.

**Catatan seri & bobot jarak:**
Jika seri, pilih $k$ ganjil atau **weighted vote**:

* Bobot tetangga-$i$: $w_i = \frac{1}{d_i + \epsilon}$ (semakin dekat, bobot semakin besar).

---

## 2) Naive Bayes (NB)

### 2.1 Teorema Bayes & asumsi “Naive”

* Inti Bayes: $P(y\mid X) \propto P(X\mid y)P(y)$.
* Asumsi “Naive” (independensi bersyarat fitur): $P(X\mid y) = \prod_i P(x_i\mid y)$.
* Pilih kelas dengan skor posterior terbesar.

**Varian umum:**

* **Multinomial / Bernoulli NB**: untuk **teks** (hitung kata/biner).
* **Gaussian NB**: untuk **fitur numerik** yang diasumsikan $\sim$ Normal per kelas.

### 2.2 Multinomial NB — contoh hitung manual (teks mini)

**Kelas:** Spam (S) vs Ham (H).
**Dokumen latih (sangat kecil):**

* Spam:
  S1: “gratis hadiah”
  S2: “gratis kupon”
  S3: “hadiah besar gratis”
* Ham:
  H1: “rapat besok”
  H2: “tugas besok dikumpulkan”
  H3: “jadwal kuliah besok”

**Langkah 1 — Prior kelas**

* Jumlah dokumen: 3 Spam, 3 Ham → $P(S)=3/6=0.5$, $P(H)=0.5$

**Langkah 2 — Frekuensi token per kelas (hanya hitung kata):**

* Spam tokens: `gratis(3)`, `hadiah(2)`, `kupon(1)`, `besar(1)` → total $6$.
* Ham tokens: `besok(3)`, `rapat(1)`, `tugas(1)`, `dikumpulkan(1)`, `jadwal(1)`, `kuliah(1)` → total $8$.
* Kamus unik $V$: 10 kata → $|V|=10$.

**Langkah 3 — Laplace smoothing ($\alpha=1$):**

* $P(w\mid S)=\dfrac{\text{count}_S(w)+\alpha}{\text{total}_S+\alpha |V|}$
* $P(w\mid H)=\dfrac{\text{count}_H(w)+\alpha}{\text{total}_H+\alpha |V|}$

Dengan $\text{total}_S=6$, $\text{total}_H=8$, $|V|=10$, $\alpha=1$:

* $P(\text{gratis}\mid S)=\dfrac{3+1}{6+10}=\dfrac{4}{16}=0.25$
* $P(\text{hadiah}\mid S)=\dfrac{2+1}{16}=\dfrac{3}{16}=0.1875$
* $P(\text{gratis}\mid H)=\dfrac{0+1}{8+10}=\dfrac{1}{18}\approx 0.0556$
* $P(\text{hadiah}\mid H)=\dfrac{0+1}{18}\approx 0.0556$

**Langkah 4 — Skor kalimat uji “gratis hadiah”:**

* Skor Spam (abaikan konstanta normalisasi):  
  $$\text{Skor}_S \propto P(S)\cdot P(\text{gratis}\mid S)\cdot P(\text{hadiah}\mid S)$$
  $$\text{Skor}_S = 0.5 \times 0.25 \times 0.1875 = 0.0234375$$

* Skor Ham:  
  $$\text{Skor}_H \propto P(H)\cdot P(\text{gratis}\mid H)\cdot P(\text{hadiah}\mid H)$$
  $$\text{Skor}_H = 0.5 \times 0.0556 \times 0.0556 \approx 0.001543$$

Bandingkan $0.0234$ vs $0.00154$ → Spam lebih besar → **Prediksi: Spam**.

**Tips praktis:** gunakan **log-prob** agar stabil:

* $\log P(y\mid X) \propto \log P(y) + \sum_i \log P(x_i\mid y)$

### 2.3 Gaussian NB — formula ringkas (numerik)

Untuk fitur numerik $x$ pada kelas $y$ dengan asumsi Normal $\mathcal{N}(\mu_y, \sigma_y^2)$:

* $P(x\mid y) = \dfrac{1}{\sqrt{2\pi\sigma_y^2}} \exp!\left(-\dfrac{(x-\mu_y)^2}{2\sigma_y^2}\right)$

Untuk banyak fitur (dianggap independen bersyarat):

* $\log P(y\mid X) \propto \log P(y) + \sum_{i} \log P(x_i\mid y)$

---

## 3) k-Fold Cross-Validation (CV)

### 3.1 Ide dasar

Satu split train/test bisa kebetulan “bagus/jelek”. **k-Fold CV** memberi perkiraan stabil:

* Bagi data jadi **$k$ fold** berukuran mirip.
* Ulangi $k$ kali: **1 fold** jadi **validasi**, **$k-1$ fold** jadi **latih**.
* Ambil **rata-rata** metrik antar fold.

**Klasifikasi:** gunakan **Stratified k-Fold** agar proporsi kelas tiap fold mirip.

### 3.2 Contoh hitung manual (k=3)

Misal 9 contoh (indeks 0…8). Bagi 3 fold:

* Fold 1: validasi = {0,1,2}, latih = {3,4,5,6,7,8}
* Fold 2: validasi = {3,4,5}, latih = {0,1,2,6,7,8}
* Fold 3: validasi = {6,7,8}, latih = {0,1,2,3,4,5}

Misal akurasi per fold:

* $s_1=0.80$, $s_2=0.75$, $s_3=0.85$

**Rata-rata (skor CV):**

* $\bar{s}=\dfrac{s_1+s_2+s_3}{3}=\dfrac{0.80+0.75+0.85}{3}=0.80$

**Simpangan baku (opsional):**

* $\sigma=\sqrt{\dfrac{(s_1-\bar{s})^2+(s_2-\bar{s})^2+(s_3-\bar{s})^2}{3}}$
* $(0.80-0.80)^2=0$
* $(0.75-0.80)^2=0.0025$
* $(0.85-0.80)^2=0.0025$
* Jumlah $=0.005$ → $\sigma=\sqrt{0.005}\approx 0.0707$

Artinya: rata-rata 0.80 dengan variasi sekitar $\pm 0.07$ di antara fold.

### 3.3 Memilih $k$

* Umum: $k=5$ atau $k=10$.
* Data kecil: $k$ bisa lebih besar, tapi hati-hati saat tuning hyperparameter.
* Data berkelompok (user sama, dsb.): gunakan **GroupKFold** agar tidak bocor antar fold.

---

## 4) Do & Don’t singkat

**Do**

* kNN: **scaling** fitur numerik; coba beberapa $k$ (3/5/7); boleh **weighted vote**.
* NB: pakai **Laplace smoothing** ($\alpha=1$) untuk teks; gunakan **log-prob**.
* CV: gunakan **Stratified** untuk klasifikasi; laporkan **mean ± std**.

**Don’t**

* kNN: jangan gabung fitur berskala jauh tanpa normalisasi.
* NB: jangan biarkan probabilitas kata nol (tanpa smoothing → skor nol).
* CV: jangan praproses di luar skema CV (hindari **leakage**); pakai **Pipeline**.

---

## 5) Latihan cepat (manual)

1. Ubah titik uji kNN menjadi $T=(4,4)$, hitung jarak ke A–F, urutkan, dan tentukan prediksi untuk $k=3$ dan $k=5$.
2. NB: uji kalimat “**gratis kupon**”. Hitung $P(\text{gratis}\mid S)$, $P(\text{kupon}\mid S)$, $P(\text{gratis}\mid H)$, $P(\text{kupon}\mid H)$, lalu bandingkan skor Spam vs Ham.
3. CV: buat k-Fold manual untuk 8 contoh dengan $k=4$, tentukan indeks fold dan rata-rata akurasi jika skor per fold $= {0.70, 0.75, 0.80, 0.85}$.

---
