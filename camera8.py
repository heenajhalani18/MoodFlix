import streamlit as st
import requests
import time
import logging
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.models import model_from_json

# TMDB API Key and URL
api_key = 'b4c3fd4bd79a7a30f43668f17e0d25bb'
base_url = 'https://api.themoviedb.org/3'

# Emotion dictionary for predictions
emotion_dict = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Neutral",
    4: "Sad",
    5: "Surprised"
}

# Function to fetch movies with retry logic
def fetch_with_retries(url, retries=5, delay=10):
    for i in range(retries):
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.debug(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                st.error("Failed to fetch data from TMDB after multiple attempts.")
                raise e

# Function to fetch TMDB genres
def fetch_genres():
    genres_url = f"{base_url}/genre/movie/list?api_key={api_key}"
    response = fetch_with_retries(genres_url)
    if response.status_code == 200:
        return {genre['id']: genre['name'] for genre in response.json()['genres']}
    else:
        st.error("Failed to fetch genres.")
        return {}

# Function to display movies with a sleek design
# Function to display movies with a sleek design and hover effects
# Function to display movies with a sleek design and hover effects
def display_movies(movies):
    if not movies:
        st.warning("No movies found.")
        return

    num_columns = 4  # Number of columns for proper alignment
    cols = st.columns(num_columns)

    for i, movie in enumerate(movies):
        with cols[i % num_columns]:  # Distribute movies evenly across columns
            st.markdown("<div class='movie-container'>", unsafe_allow_html=True)

            # Movie poster image with hover effect
            if movie.get('poster_path'):  # Ensure the poster exists
                st.markdown(
                    f"""
                    <div class='movie-poster'>
                        <img src="https://image.tmdb.org/t/p/w500{movie['poster_path']}" width="200" class="poster-img">
                        <div class='movie-synopsis'>{movie.get('overview', 'No synopsis available.')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='movie-poster'>
                        <img src="https://via.placeholder.com/200x300?text=No+Image" width="200" class="poster-img">
                        <div class='movie-synopsis'>No synopsis available.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Movie title and link to TMDB page
            st.markdown(f"<div class='movie-title'>{movie.get('title', 'Unknown')}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<p class='see-more'><a href='https://www.themoviedb.org/movie/{movie['id']}' target='_blank'>See More</a></p>",
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

# Add the hover effects CSS to the page
st.markdown(
    """
    <style>
        .movie-container {
            text-align: center;
            margin-bottom: 20px;
        }

        .movie-poster {
            position: relative;
            display: inline-block;
            overflow: hidden; /* Hide content that overflows */
            border-radius: 10px;
        }

        .poster-img {
            border-radius: 10px;
            transition: transform 0.3s ease;
        }

        .movie-poster:hover .poster-img {
            transform: scale(1.05);
        }

        .movie-synopsis {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px;
            font-size: 12px;
            text-align: left;
            height: 100%; /* Ensure the synopsis fits within the poster height */
            overflow-y: auto; /* Allow scrolling if synopsis is too long */
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease;
        }

        .movie-poster:hover .movie-synopsis {
            opacity: 1;
            visibility: visible;
        }

        .movie-title {
            font-size: 16px;
            font-weight: bold;
            color: #fff;
            margin-top: 10px;
            text-align: center;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .see-more {
            font-size: 14px;
            margin-top: 10px;
            text-align: center;
        }

        .see-more a {
            color: #2196F3;
            text-decoration: none;
        }

        .see-more a:hover {
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# Function for emotion detection
def detect_emotion(img):
    try:
        with open(r"emotion_model_equal.json", 'r') as json_file:
            loaded_model_json = json_file.read()
        emotion_model = model_from_json(loaded_model_json)
        emotion_model.load_weights(r"emotion_model_equal.weights.h5")

        frame = cv2.resize(img, (1280, 720))
        face_detector = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        maxindex = None

        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))

        return maxindex
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Initialize session states
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"
if 'on_search_page' not in st.session_state:
    st.session_state.on_search_page = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = []

# Fetch genres on startup
tmdb_genres = fetch_genres()
mood_to_genre = {
    "Angry": 28,       # Action
    "Fear": 27,        # Horror
    "Happy": 35,       # Comedy
    "Neutral": 18,     # Drama
    "Sad": 10749,      # Romance
    "Surprised": 12    # Adventure
}

# Sidebar with options
st.sidebar.header("Options")
if st.sidebar.button("Home"):
    st.session_state.current_page = "home"
if st.sidebar.button("Search Movies"):
    st.session_state.current_page = "search"
if st.sidebar.button("Detect Mood"):
    st.session_state.current_page = "mood_detection"

# Home Page - Display Popular Movies
if st.session_state.current_page == "home":
    st.title("MOODFLIX")
    popular_movies_url = f"{base_url}/movie/popular?api_key={api_key}"
    response = fetch_with_retries(popular_movies_url)
    if response.status_code == 200:
        st.header("Popular Movies")
        display_movies(response.json()['results'])

# Search Page
elif st.session_state.current_page == "search":
    search_query = st.text_input("Search for Movies")
    if st.button("Search"):
        if search_query:
            search_url = f"{base_url}/search/movie?api_key={api_key}&query={search_query}"
            response = fetch_with_retries(search_url)
            if response.status_code == 200:
                st.session_state.on_search_page = True
                st.session_state.search_results = response.json()['results']
            else:
                st.warning("No results found.")
        else:
            st.warning("Please enter a movie name to search.")
    
    if st.session_state.on_search_page:
        st.header("Search Results")
        display_movies(st.session_state.search_results)

# Mood Detection Page
elif st.session_state.current_page == "mood_detection":
    st.header("Mood-Based Movie Recommendations")
    img_file_buffer = st.camera_input("Capture an image to detect your mood")
    if img_file_buffer:
        image = Image.open(img_file_buffer)
        cv2_img = np.array(image)
        emotion_id = detect_emotion(cv2_img)
        
        if emotion_id is not None:
            detected_mood = emotion_dict[emotion_id]
            st.subheader(f"Detected Mood: {detected_mood}")
            if st.button("Show Movies for This Mood"):
                st.session_state.current_page = "recommended_movies"
                st.session_state.detected_mood = detected_mood
        else:
            st.warning("Could not detect mood.")

# Recommended Movies Page
elif st.session_state.current_page == "recommended_movies":
    detected_mood = st.session_state.detected_mood
    genre_id = mood_to_genre.get(detected_mood, 18)
    movies_url = f"{base_url}/discover/movie?api_key={api_key}&with_genres={genre_id}"
    movies_response = fetch_with_retries(movies_url)
    if movies_response.status_code == 200:
        st.header(f"Recommended Movies for {detected_mood}")
        display_movies(movies_response.json()['results'])