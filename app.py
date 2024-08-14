import streamlit as st
import pickle
import pandas as pd

# Load the similarity matrix
with open('similarity.pkl', 'rb') as file:
    similarity_matrix = pickle.load(file)

with open('movie.pkl', 'rb') as file:
    movie_list = pickle.load(file)

with open('df.pkl', 'rb') as file:
    df = pickle.load(file)


def recommend(i, similarity_matrix):
    movie_similarity_scores = similarity_matrix[i]

    # Get the indices of the top N most similar movies (excluding the movie itself)
    similar_indices = movie_similarity_scores.argsort()[-5 - 1:-1][::-1]

    # Return the recommended movies
    ans= list(df.iloc[similar_indices][0])
    return ans


# Create a DataFrame for movie_list
df_movies = pd.DataFrame(movie_list, columns=['title'])

# Streamlit app
st.title('Movie Recommendation System')

# Dropdown to select a movie
selected_movie = st.selectbox('Choose a movie:', movie_list)

# Find the index of the selected movie
selected_movie_index = df_movies[df_movies['title'] == selected_movie].index[0]

if st.button('Recommend'):
    # Get recommendations
    ans=recommend(selected_movie_index,similarity_matrix)
    for i in range(5):
        st.write(ans[i])