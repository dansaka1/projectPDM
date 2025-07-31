import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load model dan data
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('movie_titles.pkl', 'rb') as f:
    movie_titles = pickle.load(f)

df = pd.read_csv('film_dataset_cleaned.csv')

# Ambil semua genre unik
genre_list = set()
for genres in df['listed_in']:
    for genre in genres.split(', '):
        genre_list.add(genre)
genre_list = sorted(list(genre_list))

# Fungsi untuk memberikan rekomendasi
def get_recommendations(title, selected_genres, top_n=5):
    if title not in movie_titles:
        return pd.DataFrame()
    
    idx = movie_titles.index(title)
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similar_indices = cosine_sim.argsort()[::-1]

    # Filter hasil berdasarkan genre
    filtered_indices = []
    for i in similar_indices:
        film_genres = df.iloc[i]['listed_in'].split(', ')
        if any(genre in film_genres for genre in selected_genres):
            filtered_indices.append(i)
        if len(filtered_indices) >= top_n + 1:  # +1 untuk menghindari film itu sendiri
            break

    # Hapus film input dari hasil
    filtered_indices = [i for i in filtered_indices if i != idx]

    return df.iloc[filtered_indices][['title', 'listed_in', 'description']]

# ========================== UI STREAMLIT ==========================

st.set_page_config(page_title="Rekomendasi Film Netflix", page_icon="🎬")

st.markdown("<h1 style='text-align: center; color: #e50914;'>🎬 Rekomendasi Film Netflix</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Berdasarkan Beberapa Kategori Menggunakan Algoritma Content-Based Filtering</h4>", unsafe_allow_html=True)
st.markdown("---")

# Deskripsi
with st.expander("📌 Tentang Proyek Ini"):
    st.write("""
    Aplikasi ini merupakan sistem rekomendasi film yang dirancang menggunakan algoritma **Content-Based Filtering**.
    Sistem akan merekomendasikan film yang mirip dengan film pilihan pengguna berdasarkan **deskripsi dan kategori (genre)** menggunakan **metode TF-IDF dan Cosine Similarity**.
    """)

# Input pengguna
st.subheader("1. Pilih Film Favoritmu")
film_input = st.selectbox("Judul Film:", movie_titles)

st.subheader("2. Pilih Beberapa Kategori (Genre)")
selected_genres = st.multiselect("Kategori:", genre_list)

# Tampilkan hasil rekomendasi
if st.button("🎯 Tampilkan Rekomendasi"):
    if not selected_genres:
        st.warning("⚠️ Silakan pilih setidaknya satu kategori.")
    else:
        results = get_recommendations(film_input, selected_genres)
        if results.empty:
            st.error("😕 Maaf, tidak ditemukan film serupa sesuai genre yang dipilih.")
        else:
            st.markdown("## 🎥 Rekomendasi Film:")
            for idx, row in results.iterrows():
                st.markdown(f"### 🎞️ {row['title']}")
                st.write(f"**Genre:** {row['listed_in']}")
                st.write(f"**Deskripsi:** {row['description']}")
                st.markdown("---")

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: small; color: gray;'>
        © 2025 | Proyek Data Mining - Sistem Rekomendasi Film Netflix
    </div>
""", unsafe_allow_html=True)
