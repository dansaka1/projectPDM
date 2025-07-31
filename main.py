import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ======================= LOAD ============================
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('movie_titles.pkl', 'rb') as f:
    movie_titles = pickle.load(f)

df = pd.read_csv('film_dataset_cleaned.csv')

# ===================== CLEAN DATA ========================
df['duration_minutes'] = df['duration'].str.extract('(\d+)').astype(float)

# Genre
genre_list = sorted({genre.strip() for genres in df['listed_in'] for genre in genres.split(',')})
year_list = sorted(df['release_year'].dropna().unique().astype(int), reverse=True)
age_rating_list = sorted(df['age_rating'].dropna().astype(str).unique())

# Durasi filter
durasi_kategori = {
    "Kurang dari 1 jam": lambda x: x < 60,
    "Sekitar 1 jam": lambda x: 55 <= x <= 65,
    "1-2 jam": lambda x: 60 < x <= 120,
    "Lebih dari 2 jam": lambda x: x > 120,
}

# ===================== FUNGSI ============================
def get_recommendations(title, genres, years, ages, durasi_filters, top_n=5):
    if title not in movie_titles:
        return pd.DataFrame()

    idx = movie_titles.index(title)
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1]

    filtered = []
    for i in similar_indices:
        if i == idx: continue
        row = df.iloc[i]

        # Filter kombinasi
        genre_match = any(g in row['listed_in'].split(', ') for g in genres) if genres else True
        year_match = row['release_year'] in years if years else True
        age_match = str(row['age_rating']) in ages if ages else True
        durasi_val = row.get('duration_minutes', 0)
        durasi_match = any(rule(durasi_val) for rule in durasi_filters.values()) if durasi_filters else True

        if genre_match and year_match and age_match and durasi_match:
            filtered.append(i)

        if len(filtered) >= top_n:
            break

    return df.iloc[filtered][['title', 'listed_in', 'release_year', 'age_rating', 'duration', 'description', 'image_url']] if 'image_url' in df.columns else df.iloc[filtered]

# ===================== UI/UX =============================
st.set_page_config("Rekomendasi Film Netflix", "🎬", layout="wide")

st.markdown("<h1 style='text-align:center; color:#e50914;'>🎬 Rekomendasi Film Netflix</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center; color:gray;'>Berbasis Content-Based Filtering</h5>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("ℹ️ Tentang Aplikasi"):
    st.write("""
    Sistem ini menyarankan film serupa berdasarkan film yang kamu pilih. Algoritma yang digunakan:
    - **TF-IDF**: Mengubah deskripsi menjadi representasi numerik.
    - **Cosine Similarity**: Mengukur kemiripan antara film.
    - Disaring berdasarkan beberapa kategori: Genre, Tahun, Usia Penonton, dan Durasi.
    """)

st.subheader("1. Pilih Film Utama")
film_input = st.selectbox("Judul Film:", movie_titles)

st.subheader("2. Filter Tambahan")
col1, col2 = st.columns(2)

with col1:
    selected_genres = st.multiselect("🎭 Genre:", genre_list)
    selected_years = st.multiselect("📅 Tahun Rilis:", year_list)

with col2:
    selected_ages = st.multiselect("🔞 Usia Penonton:", age_rating_list)
    selected_durations = st.multiselect("⏱️ Durasi:", list(durasi_kategori.keys()))
    durasi_rules = {label: durasi_kategori[label] for label in selected_durations}

if st.button("🎯 Cari Rekomendasi"):
    results = get_recommendations(film_input, selected_genres, selected_years, selected_ages, durasi_rules)

    if results.empty:
        st.warning("⚠️ Tidak ditemukan film dengan kriteria tersebut.")
    else:
        st.success("Berikut adalah rekomendasi film untukmu:")
        for _, row in results.iterrows():
            with st.container():
                st.markdown(f"### 🎞️ {row['title']}")
                cols = st.columns([1, 3])
                if 'image_url' in row and pd.notna(row['image_url']):
                    cols[0].image(row['image_url'], width=150)
                cols[1].markdown(f"**Genre:** {row['listed_in']}")
                cols[1].markdown(f"**Tahun Rilis:** {int(row['release_year'])}")
                cols[1].markdown(f"**Rating Usia:** {row['age_rating']}")
                cols[1].markdown(f"**Durasi:** {row['duration']}")
                cols[1].markdown(f"**Deskripsi:** {row['description']}")
                st.markdown("---")

# Footer
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align:center; color:gray; font-size:small;">
    © 2025 | Proyek Data Mining - Sistem Rekomendasi Film Netflix
</div>
""", unsafe_allow_html=True)
