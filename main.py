import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ====================== LOAD MODEL DAN DATA ============================
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('movie_titles.pkl', 'rb') as f:
    movie_titles = pickle.load(f)

df = pd.read_csv('film_dataset_cleaned.csv')

# ======================== PERSIAPAN FILTER =============================
# Genre
genre_list = sorted({genre.strip() for genres in df['listed_in'] for genre in genres.split(',')})

# Tahun Rilis
year_list = sorted(df['release_year'].dropna().unique().astype(int), reverse=True)

# Rating Usia
rating_list = sorted(df['rating'].dropna().unique())

# ===================== FUNGSI REKOMENDASI ==============================
def get_recommendations(title, selected_genres, selected_years, selected_ratings, top_n=5):
    if title not in movie_titles:
        return pd.DataFrame()

    idx = movie_titles.index(title)
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1]

    # Filter berdasarkan kategori yang dipilih
    filtered_indices = []
    for i in similar_indices:
        row = df.iloc[i]

        # Genre check
        film_genres = row['listed_in'].split(', ')
        genre_match = any(genre in film_genres for genre in selected_genres) if selected_genres else True

        # Tahun check
        year_match = row['release_year'] in selected_years if selected_years else True

        # Rating check
        rating_match = row['rating'] in selected_ratings if selected_ratings else True

        if genre_match and year_match and rating_match:
            filtered_indices.append(i)

        if len(filtered_indices) >= top_n + 1:  # +1 karena input film sendiri masuk juga
            break

    # Hapus film input dari hasil
    filtered_indices = [i for i in filtered_indices if i != idx]

    return df.iloc[filtered_indices][['title', 'listed_in', 'release_year', 'rating', 'description']]

# ======================= UI STREAMLIT ===============================
st.set_page_config(page_title="Rekomendasi Film Netflix", page_icon="🎬")

st.markdown("<h1 style='text-align: center; color: #e50914;'>🎬 Rekomendasi Film Netflix</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Berdasarkan Beberapa Kategori Menggunakan Algoritma Content-Based Filtering</h4>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("📌 Tentang Proyek Ini"):
    st.write("""
    Aplikasi ini merupakan implementasi dari algoritma **Content-Based Filtering** untuk memberikan rekomendasi film Netflix. 
    Rekomendasi didasarkan pada kesamaan deskripsi film dan disaring berdasarkan:
    - **Genre**
    - **Tahun Rilis**
    - **Usia Penonton (Rating)**  
    """)
    st.info("Algoritma yang digunakan: TF-IDF + Cosine Similarity")

# ===================== INPUT PENGGUNA ============================
st.subheader("1. Pilih Film Favoritmu")
film_input = st.selectbox("Judul Film:", movie_titles)

st.subheader("2. Filter Kategori:")
col1, col2 = st.columns(2)

with col1:
    selected_genres = st.multiselect("🎭 Genre:", genre_list)

with col2:
    selected_years = st.multiselect("📅 Tahun Rilis:", year_list)

selected_ratings = st.multiselect("🔞 Rating Usia Penonton:", rating_list)

# ===================== TOMBOL DAN HASIL ==========================
if st.button("🎯 Tampilkan Rekomendasi"):
    results = get_recommendations(film_input, selected_genres, selected_years, selected_ratings)
    if results.empty:
        st.warning("😕 Tidak ditemukan film serupa yang cocok dengan kategori yang dipilih.")
    else:
        st.markdown("## 🎥 Rekomendasi Film:")
        for _, row in results.iterrows():
            st.markdown(f"### 🎞️ {row['title']}")
            st.write(f"**Genre:** {row['listed_in']}")
            st.write(f"**Tahun Rilis:** {int(row['release_year'])}")
            st.write(f"**Rating Usia:** {row['rating']}")
            st.write(f"**Deskripsi:** {row['description']}")
            st.markdown("---")

# ===================== FOOTER ==========================
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: small; color: gray;'>
        © 2025 | Proyek Data Mining - Sistem Rekomendasi Film Netflix
    </div>
""", unsafe_allow_html=True)
