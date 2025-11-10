---
marp: true
title: Teks Klasifikasi — BoW, TF-IDF, & scikit-learn Pipeline
description: Dasar representasi teks untuk klasifikasi + praktik sklearn
paginate: true
math: katex
style: |
  section { font-size: 22px; }
  h1,h2,h3 { letter-spacing: .2px }
  .muted { color:#666 }
  .pill { display:inline-block; padding:.2em .6em; border-radius:999px; background:#eef; font-size:85% }
---

# BoW • TF-IDF • Pipeline (sklearn)

**Target**
- Memahami **Bag-of-Words (BoW)** & **TF-IDF**
- Menyusun **pipeline teks** scikit-learn
- Latihan klasifikasi + evaluasi dasar

---

## Mengapa perlu representasi?

Model klasik (LogReg, SVM, NB) butuh **fitur numerik**.  
Teks → **vektor angka** via:
- **BoW**: hitung frekuensi kata
- **TF-IDF**: frekuensi × *keunikan* kata

> Intinya: kalimat diubah jadi **fitur** yang bisa dipelajari model.

---

## Bag-of-Words (BoW) — definisi

Bangun kosakata global, setiap dokumen menjadi vektor frekuensi.

$$
\text{BoW}(d) = [\, f(w_1,d),\; f(w_2,d),\; \dots,\; f(w_V,d) \,]
$$

Keterangan:  
- \(f(w_i,d)\): jumlah kemunculan kata \(w_i\) pada dokumen \(d\)  
- \(V\): ukuran kosakata

---

## BoW — contoh kecil

Kalimat:  
d₁ = “barang cepat sampai”  
d₂ = “pengiriman cepat dan rapi”

Kosakata (urut abjad): `barang, cepat, dan, pengiriman, rapi, sampai`

$$
\text{BoW}(d_1) = [1,\;1,\;0,\;0,\;0,\;1]
$$

$$
\text{BoW}(d_2) = [0,\;1,\;1,\;1,\;1,\;0]
$$

---

## TF-IDF — rumus

**Term Frequency (TF) dokumen \(d\) untuk kata \(t\):**

$$
\mathrm{tf}(t,d) = \frac{f_{t,d}}{\sum_{w} f_{w,d}}
$$

**Inverse Document Frequency (IDF):**

$$
\mathrm{idf}(t) = \log \frac{N}{1 + \mathrm{df}(t)}
$$

**Skor TF-IDF:**

$$
\mathrm{tfidf}(t,d) = \mathrm{tf}(t,d)\cdot \mathrm{idf}(t)
$$

---

## TF-IDF — intuisi cepat

- Kata **terlalu umum** (muncul di banyak dokumen) ⟶ **IDF kecil**  
- Kata **spesifik** (jarang muncul) ⟶ **IDF besar**  
- Hasil: fitur menonjolkan kata **informatif** untuk klasifikasi

---

## N-gram (konteks pendek)

- Unigram: kata tunggal → “bagus”  
- Bigram: dua kata → “tidak bagus”  
- Trigram: tiga kata → “sangat tidak bagus”

**Catatan**: negasi sering butuh **bigram** agar makna tidak hilang.

---

## Pra-proses ringkas

- Lowercase
- (Opsional) hapus URL/emoji
- Stopwords **on/off** → uji A/B (tidak selalu membantu)
- Lemma/Stemming **opsional**
- Hindari membersihkan berlebihan yang menghapus sinyal

---

## Pipeline sklearn — konsep anti-leakage

Rangkai langkah end-to-end: vektorisasi → model → evaluasi.

$$
\text{Pipeline}=\big[\ \text{Vectorizer}\ \rightarrow\ \text{Classifier}\ \big]
$$

- **Split**: train/test (stratified)  
- **fit** hanya di **train**, evaluasi di **test**  
- Mudah **GridSearchCV** untuk tuning

---

## Demo — BoW + MultinomialNB

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

data = pd.DataFrame({
  "text": [
    "barang cepat sampai", "pengiriman lambat sekali",
    "layanan ramah dan cepat", "kemasan rusak parah",
    "sangat puas mantap", "tidak puas kecewa"
  ],
  "label": ["positif","negatif","positif","negatif","positif","negatif"]
})

X_tr, X_te, y_tr, y_te = train_test_split(
  data["text"], data["label"], test_size=0.33, stratify=data["label"], random_state=42
)

pipe_bow_nb = Pipeline([
  ("bow", CountVectorizer(ngram_range=(1,2), min_df=1)),
  ("clf", MultinomialNB())
])

pipe_bow_nb.fit(X_tr, y_tr)
print(classification_report(y_te, pipe_bow_nb.predict(X_te)))
````

---

## Demo — TF-IDF + Logistic Regression

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

pipe_tfidf_lr = Pipeline([
  ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
  ("clf", LogisticRegression(max_iter=1000))
])

pipe_tfidf_lr.fit(X_tr, y_tr)
print(classification_report(y_te, pipe_tfidf_lr.predict(X_te)))
```

**Catatan**: LogReg kuat sebagai baseline teks; tuning **C** (regularisasi).

---

## Evaluasi yang dibaca

* **Accuracy**: proporsi benar
* **Precision/Recall/F1** per kelas (penting untuk imbalance)
* **Confusion Matrix**: salahnya di mana

---

## Grid Search ringan (opsional)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
  "tfidf__ngram_range": [(1,1), (1,2)],
  "tfidf__min_df": [1, 2, 3],
  "clf__C": [0.1, 1.0, 3.0]
}

grid = GridSearchCV(pipe_tfidf_lr, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
grid.fit(X_tr, y_tr)

print("Best params:", grid.best_params_)
print("Best cv score:", grid.best_score_)
print(classification_report(y_te, grid.best_estimator_.predict(X_te)))
```

---

## Jebakan umum & praktik baik

* **Split dulu**, lalu **fit/transform** hanya di **train** → hindari **leakage**
* Set **random_state** untuk replikasi
* Uji **unigram vs bigram** (negasi)
* Coba **stopwords on/off** (uji empiris)
* Batasi dimensi dengan **max_features** bila perlu
* Tangani **teks kosong**/noise di awal

---

## Template umum (siap pakai)

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# df: kolom "text", "label"
X_tr, X_te, y_tr, y_te = train_test_split(
  df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

pipe = Pipeline([
  ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2)),
  ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(X_tr, y_tr)
y_pr = pipe.predict(X_te)
print(classification_report(y_te, y_pr))
```

---

## Checklist tugas

* [ ] Dataset teks + label siap
* [ ] Split stratified (train/test)
* [ ] Pipeline (BoW/TF-IDF + model) berjalan
* [ ] Laporan **accuracy**, **macro-F1**, **Confusion Matrix**
* [ ] Catat eksperimen: n-gram, stopwords, min_df, model
