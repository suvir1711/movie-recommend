import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# set app config
st.set_page_config(page_title="MRS", page_icon="🍿", layout="wide")    
st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://www.pexels.com/photo/white-clouds-on-blue-sky-19670/"); 
                     background-attachment: fixed;
                     base: dark;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

# Load models and MovieDB
df = joblib.load('models/movie_db.df')
tfidf_matrix = joblib.load('models/tfidf_mat.tf')
tfidf = joblib.load('models/vectorizer.tf')
cos_mat = joblib.load('models/cos_mat.mt')

# define functions
def get_keywords_recommendations(keywords):
    
    keywords = keywords.split()
    keywords = " ".join(keywords)    
    # transform the string to vector representation
    key_tfidf = tfidf.transform([keywords]) 
    # compute cosine similarity    
    result = cosine_similarity(key_tfidf, tfidf_matrix)
    # sort top n similar movies   
    similar_key_movies = sorted(list(enumerate(result[0])), reverse=True, key=lambda x: x[1])
    # extract names from dataframe and return movie names
    recomm = []
    for i in similar_key_movies[1:6]:
        recomm.append(df.iloc[i[0]].title)
    return recomm

def get_recommendations(movie):
    
    # get index from dataframe
    index = df[df['title']== movie].index[0]    
    # sort top n similar movies     
    similar_movies = sorted(list(enumerate(cos_mat[index])), reverse=True, key=lambda x: x[1]) 
    # extract names from dataframe and return movie names
    recomm = []
    for i in similar_movies[1:6]:
        recomm.append(df.iloc[i[0]].title)
    return recomm

def fetch_poster(ids):
    posters = []
    for i in ids:    
        url = f"https://api.themoviedb.org/3/movie/{i}?api_key=93cba6645f461e73dcdf61d8dddad4de"
        data = requests.get(url)
        data = data.json()
        # Check if 'poster_path' exists in the JSON response
        if 'poster_path' in data:
            poster_path = data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            posters.append(full_path)
    return posters





# App Layout

st.title("Movie Recommendation System 🍿 🤖")
posters = 0
movies = 0

with st.sidebar:
    st.header("Get Recommendations by 👇")
    search_type = st.radio(" ", ('Movie Title', 'Keywords'))
    st.header("Source Code 📦")
    st.markdown("[GitHub Repository](https://github.com/afaqueumer/TMDBAI)")    
    

    
    

# call functions based on selectbox
if search_type == 'Movie Title': 
    st.subheader("Select Movie 🎬")   
    movie_name = st.selectbox(' ', df.title)
    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies = get_recommendations(movie_name)
            posters = fetch_poster(movies)
            st.text(movies)
            
                   
else:
    st.subheader('Enter Cast / Crew / Tags / Genre  🌟')
    keyword = st.text_input(' ', 'Christopher Nolan')
    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies = get_keywords_recommendations(keyword)
            posters = fetch_poster(movies)
            st.text(movies)
        
            


            
            
              