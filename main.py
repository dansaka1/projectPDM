import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load komponen model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('movie_titles.pkl', 'rb') as f:
    movie_titles = pickle.load(f)

# Load dataset
df = pd.read_csv('film_dataset_cleaned.csv')

# Ambil daftar genre unik
genre_list = set()
for genres in df['listed_in']:
    for genre in genres.split(', '):
        genre_list.add(genre)
genre_list = sorted(list(genre_list))

# Fungsi rekomendasi dengan filter genre dan hasil lengkap
def get_recommendations(title, selected_genres, top_n=5):
    if title not in movie_titles:
        return pd.DataFrame()

    idx = movie_titles.index(title)
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[::-1]

    filtered_indices = []
    for i in similar_indices:
        film_genres = df.iloc[i]['listed_in'].split(', ')
        if any(genre in film_genres for genre in selected_genres):
            filtered_indices.append(i)
        if len(filtered_indices) >= top_n + 1:  # +1 karena film itu sendiri akan masuk juga
            break

    # Hilangkan film input dari hasil
    filtered_indices = [i for i in filtered_indices if i != idx]

    # Kolom yang ditampilkan
    selected_columns = [
        'show_id', 'type', 'title', 'director', 'cast', 'country',
        'date_added', 'release_year', 'rating', 'duration',
        'listed_in', 'description'
    ]

    # Pastikan semua kolom tersedia di DataFrame
    selected_columns = [col for col in selected_columns if col in df.columns]

    return df.iloc[filtered_indices][selected_columns]

# UI Streamlit
st.set_page_config("Rekomendasi Film Netflix", page_icon="🎬")
st.title("🎬 Rekomendasi Film Netflix")

film_input = st.selectbox("Pilih judul film yang kamu sukai:", movie_titles)
selected_genres = st.multiselect("Pilih satu atau lebih genre:", genre_list)

if st.button("Tampilkan Rekomendasi"):
    if not selected_genres:
        st.warning("Silakan pilih setidaknya satu genre.")
    else:
        results = get_recommendations(film_input, selected_genres)
        if results.empty:
            st.warning("Tidak ada rekomendasi ditemukan.")
        else:
            for _, row in results.iterrows():
                st.markdown(f"### 🎞️ {row.get('title', 'Tanpa Judul')}")
                st.write(f"**Jenis:** {row.get('type', '-')}")
                st.write(f"**Show ID:** {row.get('show_id', '-')}")
                st.write(f"**Sutradara:** {row.get('director', '-')}")
                st.write(f"**Pemeran:** {row.get('cast', '-')}")
                st.write(f"**Negara:** {row.get('country', '-')}")
                st.write(f"**Tanggal Ditambahkan:** {row.get('date_added', '-')}")
                st.write(f"**Tahun Rilis:** {row.get('release_year', '-')}")
                st.write(f"**Peringkat (Usia):** {row.get('rating', '-')}")
                st.write(f"**Durasi:** {row.get('duration', '-')}")
                st.write(f"**Tercantum Dalam (Genre):** {row.get('listed_in', '-')}")
                st.write(f"**Deskripsi:** {row.get('description', '-')}")
                st.markdown("---")
