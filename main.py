import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load semua komponen
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('movie_titles.pkl', 'rb') as f:
    movie_titles = pickle.load(f)

df = pd.read_csv('film_dataset_cleaned.csv')

# Ambil daftar genre unik
genre_list = set()
for genres in df['listed_in']:
    for genre in genres.split(', '):
        genre_list.add(genre)
genre_list = sorted(list(genre_list))

# Fungsi rekomendasi
def get_recommendations(title, selected_genres, top_n=5):
    if title not in movie_titles:
        return pd.DataFrame()
    
    idx = movie_titles.index(title)
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Ambil indeks rekomendasi berdasarkan cosine similarity
    similar_indices = cosine_sim.argsort()[::-1]

    # Filter berdasarkan genre yang dipilih
    filtered_indices = []
    for i in similar_indices:
        film_genres = df.iloc[i]['listed_in'].split(', ')
        if any(genre in film_genres for genre in selected_genres):
            filtered_indices.append(i)
        if len(filtered_indices) >= top_n + 1:  # +1 karena film itu sendiri akan masuk juga
            break

    # Hilangkan film itu sendiri dari hasil
    filtered_indices = [i for i in filtered_indices if i != idx]

    return df.iloc[filtered_indices][['title', 'listed_in', 'description']]

# UI Streamlit
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
            for idx, row in results.iterrows():
                st.subheader(row['title'])
                st.write(f"**Genre:** {row['listed_in']}")
                st.write(f"**Deskripsi:** {row['description']}")
                st.markdown("---")
