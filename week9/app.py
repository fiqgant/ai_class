from math import log, sqrt
import numpy as np

# -----------------------------
# 1) Dataset kecil + tokenisasi
# -----------------------------
docs = [
    "barang cepat sampai",          # d1
    "pengiriman cepat dan rapi",    # d2
    "sangat lambat pengiriman",     # d3
]
doc_ids = ["d1", "d2", "d3"]

def tokenize(s: str):
    # Tokenisasi sederhana: lowercase + split spasi
    return s.lower().strip().split()

tokens_per_doc = [tokenize(d) for d in docs]

# -----------------------------
# 2) Bangun kosakata (urut abjad)
# -----------------------------
vocab = sorted({tok for toks in tokens_per_doc for tok in toks})
V = len(vocab)
term2idx = {t:i for i,t in enumerate(vocab)}

print("KOSAKATA (urut abjad) =", vocab)
print("Ukuran kosakata (V)  =", V)
print()

# -----------------------------------------
# 3) BoW: frekuensi kata per dokumen (manual)
# -----------------------------------------
def bow_vector(tokens, term2idx, V):
    vec = np.zeros(V, dtype=int)
    for t in tokens:
        if t in term2idx:
            vec[term2idx[t]] += 1
    return vec

BOW = np.vstack([bow_vector(toks, term2idx, V) for toks in tokens_per_doc])
print("Matriks BoW (baris = dokumen, kolom = kata):")
print(BOW)
print()

# --------------------------------
# 4) TF, DF, IDF (manual, definisi)
# --------------------------------
# TF(d, t) = freq(t, d) / |d|
doc_lengths = np.array([len(toks) for toks in tokens_per_doc], dtype=float)
TF = BOW / doc_lengths[:, None]

# DF(t) = jumlah dokumen yang mengandung t
DF = np.count_nonzero(BOW > 0, axis=0)

# IDF(t) = ln( N / (1 + DF(t)) )   (varian aman)
N = len(docs)
IDF = np.log(N / (1.0 + DF))

print("Panjang dokumen:", doc_lengths.tolist())
print("DF per term    :", dict(zip(vocab, DF.tolist())))
print("IDF per term   :", dict(zip(vocab, np.round(IDF, 4).tolist())))
print()

# --------------------------
# 5) TF-IDF (manual, matriks)
# --------------------------
TFIDF = TF * IDF  # broadcast kolom IDF
print("Matriks TF-IDF (dibulatkan 4 desimal):")
print(np.round(TFIDF, 4))
print()

# -----------------------------------------------------
# 6) Cosine similarity (manual) antara query dan dokumen
# -----------------------------------------------------
def cosine(u: np.ndarray, v: np.ndarray) -> float:
    num = float(np.dot(u, v))
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    return 0.0 if den == 0.0 else num / den

query = "barang cepat"
q_tokens = tokenize(query)

# vektor TF-IDF untuk query (pakai skema TF-IDF yang sama)
q_bow = bow_vector(q_tokens, term2idx, V)
q_tf  = q_bow / max(1, len(q_tokens))
q_tfidf = q_tf * IDF

print(f'Query: "{query}"')
print("q TF-IDF (dibulatkan 4 desimal):", np.round(q_tfidf, 4).tolist())

sims = [cosine(q_tfidf, TFIDF[i]) for i in range(N)]
for i, s in enumerate(sims):
    print(f"cosine(query, {doc_ids[i]}) = {s:.3f}")
print()

# --------------------------------------------
# 7) Versi scikit-learn (untuk verifikasi cepat)
# --------------------------------------------
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # BoW dengan sklearn
    cv = CountVectorizer(lowercase=True, tokenizer=str.split)  # tokenizer sederhana
    X_bow = cv.fit_transform(docs)
    print("[sklearn] vocab_ =", sorted(cv.vocabulary_.keys()))
    print("[sklearn] BoW shape:", X_bow.shape)

    # TF-IDF dengan sklearn (default idf = log((1+n)/(1+df))+1) — berbeda formula dari manual
    # Di sini kita set use_idf=True, smooth_idf=True (default), sublinear_tf=False agar standar
    tv = TfidfVectorizer(lowercase=True, tokenizer=str.split, ngram_range=(1,1))
    X_tfidf = tv.fit_transform(docs)

    # Query → TF-IDF (sklearn)
    q_tfidf_sk = tv.transform([query])
    sims_sk = cosine_similarity(q_tfidf_sk, X_tfidf).ravel()

    print("[sklearn] Cosine(query, docs) =", np.round(sims_sk, 3).tolist())
    print("\nCatatan: skor TF-IDF sklearn bisa berbeda skala/angka dibanding manual karena definisi IDF yang berbeda,")
    print("namun urutan kemiripan umumnya sejalan (dokumen paling relevan tetap sama).")

except Exception as e:
    print("\n[PERINGATAN] Bagian verifikasi sklearn dilewati karena:", e)
