# Hitungan Manual — **Decision Tree**, **Feature Engineering**, & **Grid Search (Ringan)**


---

## 1) Decision Tree — **Impurity** & **Gain** (Klasifikasi) + **MSE** (Regresi)

### 1.1 Definisi (Klasifikasi)

**Gini**

$$
\mathrm{Gini}(S) ;=; 1 - \sum_k p_k^2
$$

**Entropy**

$$
H(S) ;=; - \sum_k p_k \log_2 p_k
$$

**Impurity setelah split** ($S \to S_1,\ldots,S_m$)

$$
\mathrm{Impurity_after} ;=; \sum_{j=1}^{m} \frac{|S_j|}{|S|},\mathrm{Impurity}(S_j)
$$

**Information Gain**

$$
\mathrm{Gain} ;=; \mathrm{Impurity}(S) ;-; \mathrm{Impurity_after}
$$

---

### 1.2 Contoh Manual (Klasifikasi, Gini)

Node akar: 10 data → 4 kelas-1 (Spam), 6 kelas-0 (Ham). Kandidat split fitur biner **has_gratis**.

**Gini awal**

$$
\mathrm{Gini}(S) ;=; 1 - (0.4^2 + 0.6^2)
;=; 1 - (0.16 + 0.36) ;=; 0.48
$$

**Setelah split `has_gratis`**

* **Kiri** ($S_L$): 4 data → 3 Spam, 1 Ham
  $$
  \mathrm{Gini}(S_L) ;=; 1 - (0.75^2 + 0.25^2)
  ;=; 1 - (0.5625 + 0.0625) ;=; 0.375
  $$
* **Kanan** ($S_R$): 6 data → 1 Spam, 5 Ham
  $$
  \mathrm{Gini}(S_R) ;=; 1 - \left((\tfrac{1}{6})^2 + (\tfrac{5}{6})^2\right)
  ;\approx; 1 - (0.0278 + 0.6944) ;=; 0.2778
  $$

**Rata-rata tertimbang**

$$
\mathrm{Gini_after} ;=; \frac{4}{10}\cdot 0.375 ;+; \frac{6}{10}\cdot 0.2778
;=; 0.3167
$$

**Gain**

$$
\mathrm{Gain} ;=; 0.48 ;-; 0.3167 ;=; 0.1633
$$

---

### 1.3 Contoh Manual (Klasifikasi, Entropy — opsional)

$$
H(S) ;=; -\big(0.4 \log_2 0.4 ;+; 0.6 \log_2 0.6\big) ;\approx; 0.97095
$$

> Lakukan langkah sama: hitung $H(S_L)$ dan $H(S_R)$, lalu rata-rata tertimbang dan kurangi dari $H(S)$ untuk memperoleh $\mathrm{Gain}$ berbasis Entropy.

---

### 1.4 Contoh Manual (Regresi, MSE)

Node awal berisi target $y={2,3,4,10}$.

**MSE node awal**

$$
\bar{y} ;=; \frac{2+3+4+10}{4} ;=; 4.75
$$

$$
\mathrm{MSE}(S) ;=; \frac{(2-4.75)^2 + (3-4.75)^2 + (4-4.75)^2 + (10-4.75)^2}{4}
;=; 9.6875
$$

**Split**: $S_L={2,3,4}$, $S_R={10}$

* $S_L$:
  $$
  \bar{y}_L ;=; 3, \qquad
  \mathrm{MSE}(S_L) ;=; \frac{(2-3)^2 + (3-3)^2 + (4-3)^2}{3}
  ;=; \frac{2}{3} ;\approx; 0.6667
  $$
* $S_R$: satu nilai → $\mathrm{MSE}(S_R)=0$

**Rata-rata tertimbang**

$$
\mathrm{MSE_after} ;=; \frac{3}{4}\cdot \frac{2}{3} ;+; \frac{1}{4}\cdot 0
;=; 0.5
$$

**Reduction**

$$
\Delta \mathrm{MSE} ;=; 9.6875 ;-; 0.5 ;=; 9.1875
$$

---

## 2) Feature Engineering — **Transformasi Manual**

### 2.1 One-Hot Encoding (Kategori → Vektor Biner)

Misal `channel \in \{\mathrm{email}, \mathrm{chat}, \mathrm{web}\}`. Untuk contoh dengan `channel=chat`:

$$
[\mathrm{email}=0,; \mathrm{chat}=1,; \mathrm{web}=0]
$$

> Pohon dapat split sederhana seperti `chat == 1`.

---

### 2.2 Binning (Numerik → Kategori)

Misal `age = \{18,22,25,31,35,49\}` dan bin:

* Young: $x < 25$
* Adult: $25 \le x < 40$
* Senior: $x \ge 40$

Contoh: $x=31 \Rightarrow \mathrm{Adult}$ → One-Hot $[0,1,0]$.

---

### 2.3 Fitur Waktu **Siklik** (jam 0 ≈ 24)

Gunakan pemetaan siklik agar 23 dekat dengan 0:

$$
\sin_{\mathrm{hour}} ;=; \sin!\left(2\pi \cdot \frac{\mathrm{hour}}{24}\right),
\qquad
\cos_{\mathrm{hour}} ;=; \cos!\left(2\pi \cdot \frac{\mathrm{hour}}{24}\right)
$$

Contoh nilai:

* $\mathrm{hour}=0 \Rightarrow (0,,1)$
* $\mathrm{hour}=6 \Rightarrow (1,,0)$
* $\mathrm{hour}=12 \Rightarrow (0,,-1)$
* $\mathrm{hour}=18 \Rightarrow (-1,,0)$
* $\mathrm{hour}=23 \Rightarrow (\approx -0.2588,, \approx 0.9659)$

---

### 2.4 Interaksi Fitur (Produk Sederhana)

Tambahkan fitur:

$$
x_{1\times 2} ;=; x_1 \cdot x_2
$$

Contoh: $(x_1,x_2)=(2,5) \Rightarrow x_{1\times 2}=10$.

---

### 2.5 TF–IDF Ringkas (untuk Teks)

Definisi umum:

$$
\operatorname{tfidf}(t,d) ;=; \operatorname{tf}(t,d)\cdot \operatorname{idf}(t,D),
\qquad
\operatorname{idf}(t,D) ;=; \log!\left(\frac{|D|}{1+\operatorname{df}(t)}\right)
$$

Contoh mini: $|D|=3$, kata `gratis` muncul di 1 dokumen ⇒ $\operatorname{idf}=\log(3/2)\approx 0.4055$.
Jika $\operatorname{tf}(t,d)=2$, maka $\operatorname{tfidf}\approx 0.811$.

> Hasil TF–IDF adalah fitur numerik yang bisa di-split oleh Decision Tree.

---

## 3) Grid Search (Ringan) — **k-Fold Mean & Std Manual**

### 3.1 Rata-rata & Simpangan Baku 5-Fold

Dua kandidat parameter (A, B) dengan skor akurasi per fold:

* **Param A**: ${0.78,,0.80,,0.82,,0.79,,0.81}$
  Mean:
  $$
  \bar{s}_A ;=; \frac{4.00}{5} ;=; 0.80
  $$
  Std (populasi):
  $$
  \sigma_A ;=; \sqrt{\frac{(0.78-0.80)^2+(0.80-0.80)^2+(0.82-0.80)^2+(0.79-0.80)^2+(0.81-0.80)^2}{5}}
  ;=; \sqrt{\frac{0.001}{5}} ;\approx; 0.0141
  $$

* **Param B**: ${0.80,,0.81,,0.83,,0.80,,0.82}$
  Mean:
  $$
  \bar{s}_B ;=; \frac{4.06}{5} ;=; 0.812
  $$

**Keputusan sederhana**: pilih **Param B** karena mean lebih tinggi.
Jika selisih tipis dan $\sigma$ besar, pilih parameter **lebih sederhana**.

---

### 3.2 Tabel Grid Kecil (Contoh Tanpa Kode)

Tuning **Decision Tree**:

* $\mathrm{max_depth} \in {3,5,7}$
* $\mathrm{min_samples_leaf} \in {1,3}$

| max_depth | min_samples_leaf | mean CV (F1) |
| --------: | ---------------: | -----------: |
|         3 |                1 |         0.78 |
|         3 |                3 |         0.77 |
|         5 |                1 |         0.81 |
|         5 |                3 |         0.80 |
|         7 |                1 |         0.81 |
|         7 |                3 |         0.79 |

Jika seri (mis. 0.81 vs 0.81), pilih konfigurasi **lebih sederhana** (kedalaman lebih kecil).

---

## 4) Do & Don’t (Ringkas)

**Do**

* Kontrol kompleksitas pohon: `max_depth`, `min_samples_leaf`, pruning.
* FE sederhana dulu (one-hot, binning, siklik, interaksi kecil).
* Grid kecil + k-fold; laporkan **mean ± std**.

**Don’t**

* **Leakage**: jangan fit transformasi pada seluruh data sebelum CV/split.
* Grid terlalu besar tanpa alasan (waktu lama, rawan overfit CV).
* FE berlebihan di data kecil (dimensi meledak, noise).

---

## 5) Latihan Manual

1. **DT (Gini):** Ubah split jadi $S_L={2\ \mathrm{Spam},2\ \mathrm{Ham}}$ dan $S_R={2\ \mathrm{Spam},4\ \mathrm{Ham}}$. Hitung $\mathrm{Gini_after}$ dan $\mathrm{Gain}$ dari akar ($0.48$).
2. **Regresi (MSE):** Tambahkan outlier $50$ ke node awal $y={2,3,4,10,50}$. Hitung MSE awal dan reduction untuk split $S_L={2,3,4}$, $S_R={10,50}$.
3. **Siklik:** Hitung $(\sin_{\mathrm{hour}},\cos_{\mathrm{hour}})$ untuk $\mathrm{hour}\in{0,6,12,18,23}$.
4. **Grid:** Dengan skor ${0.74,0.77,0.79,0.76,0.78}$, hitung mean & std (rumus di atas).

---
