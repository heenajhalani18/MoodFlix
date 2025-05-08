# MoodFlix
MoodFlix is a web-based movie recommendation system that detects your real-time facial emotion using CNN and suggests movies based on your current mood. It enhances user experience with personalized, emotion-aware recommendations powered by AI, OpenCV, Flask, and TMDB API.

Features:-
-Real-time emotion detection (Happy, Sad, Neutral, Angry, Surprised) using CNN & OpenCV
-Mood-to-Genre mapping for instant movie recommendations
-Content-based filtering using TF-IDF + Cosine Similarity
-"Like" movies to build a personalized "For You" section
-Search movies by title with quick access to TMDB links
-User authentication with Flask-Login & Bcrypt
-Dynamic dashboard with trending movies & real-time updates

Tech Stack:-
-Frontend: HTML/CSS, JavaScript
-Backend: Python (Flask), Flask-Login, Flask-Bcrypt
-AI/ML: CNN for facial expression recognition
-APIs: TMDB for movie data
-Libraries: OpenCV, scikit-learn, pandas, numpy

How It Works:-
1. Login to the platform
2. Capture mood via webcam → CNN model classifies emotion
3. Recommend movies based on mood & liked preferences
4. Like movies → system refines future suggestions
5. Search & explore trending movies via TMDB API
