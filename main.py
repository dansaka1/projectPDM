import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data dan model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('movie_titles.pkl', 'rb') as f:
    movie_titles = pickle.load(f)

df = pd.read_csv('film_dataset_cleaned.csv')

# Pastikan kolom durasi numerik dan bersih
df['duration_minutes'] = df['duration'].str.extract('(\d+)').astype(float)

# Genre unik
genre_list = sorted({genre.strip() for genres in df['listed_in'] for genre in genres.split(',')})
year_list = sorted(df['release_year'].dropna().unique().astype(int), reverse=True)
age_rating_list = sorted(df['age_rating'].dropna().astype(str).unique()) if 'age_rating' in df.columns else ['13', '17', '18', '21']

# Kategori durasi custom
durasi_opsi = {
    "Kurang dari 1 jam": lambda x: x < 60,
    "Sekitar 1 jam": lambda x: 55 <= x <= 65,
    "Lebih dari 1 jam": lambda x: 60 < x <= 90,
    "Sekitar 2 jam": lambda x: 90 <= x <= 130,
    "Lebih dari 2 jam": lambda x: x > 130,
}

# Fungsi rekomendasi
def get_recommendations(title, selected_genres, selected_years, selected_ages, selected_durations, top_n=5):
    if title not in movie_titles:
        return pd.DataFrame()

    idx = movie_titles.index(title)
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1]

    filtered_indices = []
    for i in similar_indices:
        if i == idx:
            continue
        row = df.iloc[i]

        # Genre filter
        film_genres = row['listed_in'].split(', ')
        genre_match = any(genre in film_genres for genre in selected_genres) if selected_genres else True

        # Tahun rilis
        year_match = row['release_year'] in selected_years if selected_years else True

        # Rating usia
        age_match = str(row.get('age_rating', '')) in selected_ages if selected_ages else True

        # Durasi
        durasi_value = row.get('duration_minutes', 0)
        durasi_match = any(condition(durasi_value) for condition in selected_durations.values()) if selected_durations else True

        if genre_match and year_match and age_match and durasi_match:
            filtered_indices.append(i)

        if len(filtered_indices) >= top_n:
            break

    return df.iloc[filtered_indices][['title', 'listed_in', 'release_year', 'age_rating', 'duration', 'description', 'image_url']] if 'image_url' in df.columns else \
           df.iloc[filtered_indices][['title', 'listed_in', 'release_year', 'age_rating', 'duration', 'description']]

# ======================= UI ========================

st.set_page_config(page_title="Rekomendasi Film", page_icon="🎬")

with st.expander("Tentang Aplikasi Ini"):
    st.write("""
    Sistem ini merekomendasikan film berdasarkan konten deskripsi dengan Content-Based Filtering.
    Pengguna dapat memfilter hasil berdasarkan beberapa kategori seperti genre, tahun rilis, usia penonton, dan durasi film.
    """)

st.selectbox("Pilih Film Favoritmu", movie_titles, key="film_input", index=0)
film_input = st.session_state.film_input

st.markdown("### Filter Kategori:")
col1, col2 = st.columns(2)

with col1:
    selected_genres = st.multiselect("Genre", genre_list)
    selected_years = st.multiselect("Tahun Rilis", year_list)

with col2:
    selected_ages = st.multiselect("Usia Penonton", age_rating_list)
    durasi_label = st.multiselect("Durasi Film", list(durasi_opsi.keys()))
    selected_durations = {label: durasi_opsi[label] for label in durasi_label}

if st.button("Tampilkan Rekomendasi"):
    results = get_recommendations(film_input, selected_genres, selected_years, selected_ages, selected_durations)
    if results.empty:
        st.error("Tidak ada film ditemukan dengan kriteria tersebut.")
    else:
        for _, row in results.iterrows():
            st.subheader(f"🎬 {row['title']}")
            if 'image_url' in row and pd.notna(row['image_url']):
                st.image(row['image_url'], use_column_width=True)
            st.markdown(f"**Genre:** {row['listed_in']}")
            st.markdown(f"**Tahun Rilis:** {int(row['release_year'])}")
            st.markdown(f"**Usia Penonton:** {row.get('age_rating', 'Tidak tersedia')}")
            st.markdown(f"**Durasi:** {row.get('duration', 'Tidak diketahui')}")
            st.markdown(f"**Deskripsi:** {row['description']}")
            st.markdown("---")
