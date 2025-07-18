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

# Fungsi rekomendasi
def get_recommendations(title, top_n=5):
    if title not in movie_titles:
        return []
    
    idx = movie_titles.index(title)
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    
    return df.iloc[similar_indices][['title', 'listed_in', 'description']]

# UI Streamlit
st.title("🎬 Rekomendasi Film Netflix")

film_input = st.selectbox("Pilih judul film yang kamu sukai:", movie_titles)

if st.button("Tampilkan Rekomendasi"):
    results = get_recommendations(film_input)
    if results.empty:
        st.warning("Tidak ada rekomendasi ditemukan.")
    else:
        for idx, row in results.iterrows():
            st.subheader(row['title'])
            st.write(f"**Genre:** {row['listed_in']}")
            st.write(f"**Deskripsi:** {row['description']}")
            st.markdown("---")
