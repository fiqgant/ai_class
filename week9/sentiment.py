import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# =========================================================
# Konfigurasi Streamlit
# =========================================================
st.set_page_config(page_title="Sentiment Analysis — CSV Auto", layout="wide")
st.title("Klasifikasi Sentiment (Positif / Negatif)")
st.caption("Dataset otomatis diunduh dari sumber publik (IMDB / Yelp CSV).")

# =========================================================
# Sidebar: konfigurasi
# =========================================================
st.sidebar.header("Konfigurasi Dataset")

dataset_choice = st.sidebar.selectbox(
    "Pilih dataset",
    ["IMDB (Movie Reviews)", "Yelp Polarity (Restaurant Reviews)"]
)

vec_choice = st.sidebar.radio("Vectorizer", ["TF-IDF", "BoW"], index=0)
model_choice = st.sidebar.radio("Model", ["LogisticRegression", "MultinomialNB"], index=0)
ng_min, ng_max = st.sidebar.select_slider("n-gram range", options=[1, 2, 3], value=(1, 2))
min_df = st.sidebar.number_input("min_df", min_value=1, max_value=10, value=2)
test_size = st.sidebar.slider("Proporsi data TEST", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42)
C = st.sidebar.number_input("C (LogReg)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
alpha = st.sidebar.number_input("alpha (MultinomialNB)", min_value=0.001, max_value=10.0, value=1.0, step=0.1)

# =========================================================
# Auto-download dataset publik
# =========================================================
@st.cache_data(show_spinner=True)
def load_public_dataset(name: str) -> pd.DataFrame:
    if name.startswith("IMDB"):
        url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
        df = pd.read_csv(url)
        # adaptasikan ke format umum
        if "tweet" in df.columns:
            df = df.rename(columns={"tweet": "text", "label": "label"})
        df = df[["text", "label"]]
        df["label"] = df["label"].replace({0: "negatif", 1: "positif"})
    else:
        url = "https://raw.githubusercontent.com/SkalskiP/ML_Text_Classification/master/data/yelp.csv"
        df = pd.read_csv(url)
        df = df.rename(columns={"text": "text", "sentiment": "label"})
        df["label"] = df["label"].replace({"negative": "negatif", "positive": "positif"})
    return df.dropna().reset_index(drop=True)

with st.spinner(f"Mengunduh dataset {dataset_choice}..."):
    df = load_public_dataset(dataset_choice)

st.subheader("1) Dataset")
st.write("Jumlah data:", len(df))
st.dataframe(df.head(10), use_container_width=True, height=250)
st.write("Distribusi label:", dict(df["label"].value_counts()))

# Split train/test
df_train, df_test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=random_state)

# =========================================================
# Model training
# =========================================================
st.subheader("2) Training & Evaluasi")

vectorizer = (
    TfidfVectorizer(lowercase=True, ngram_range=(ng_min, ng_max), min_df=min_df)
    if vec_choice == "TF-IDF"
    else CountVectorizer(lowercase=True, ngram_range=(ng_min, ng_max), min_df=min_df)
)

clf = LogisticRegression(max_iter=1000, C=C) if model_choice == "LogisticRegression" else MultinomialNB(alpha=alpha)
pipe = Pipeline([("vec", vectorizer), ("clf", clf)])

with st.spinner("Melatih model..."):
    pipe.fit(df_train["text"], df_train["label"])

y_true = df_test["label"]
y_pred = pipe.predict(df_test["text"])

L, R = st.columns(2)
with L:
    st.write("**Classification Report:**")
    st.code(classification_report(y_true, y_pred, digits=3), language="text")
with R:
    st.write("**Confusion Matrix:**")
    labels_sorted = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted)
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="black")
    fig.colorbar(im)
    st.pyplot(fig, use_container_width=True)

# =========================================================
# Inference interaktif
# =========================================================
st.subheader("3) Coba Prediksi Kalimat")
user_text = st.text_area("Masukkan kalimat ulasan:", height=100)
if st.button("Prediksi"):
    if not user_text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        pred = pipe.predict([user_text])[0]
        st.success(f"Prediksi: **{pred}**")
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba([user_text])[0]
            st.write({cls: round(float(p), 3) for cls, p in zip(pipe.classes_, probs)})
