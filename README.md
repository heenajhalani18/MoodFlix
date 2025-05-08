# MoodFlix
MoodFlix is a web-based movie recommendation system that detects your real-time facial emotion using CNN and suggests movies based on your current mood. It enhances user experience with personalized, emotion-aware recommendations powered by AI, OpenCV, Flask, and TMDB API.

Features:-
1. Real-time emotion detection (Happy, Sad, Neutral, Angry, Surprised) using CNN & OpenCV
2. Mood-to-Genre mapping for instant movie recommendations
3. Content-based filtering using TF-IDF + Cosine Similarity
4. "Like" movies to build a personalized "For You" section
5. Search movies by title with quick access to TMDB links
6. User authentication with Flask-Login & Bcrypt
7. Dynamic dashboard with trending movies & real-time updates

Tech Stack:-
1. Frontend: HTML/CSS, JavaScript
2. Backend: Python (Flask), Flask-Login, Flask-Bcrypt
3. AI/ML: CNN for facial expression recognition
4. APIs: TMDB for movie data
5. Libraries: OpenCV, scikit-learn, pandas, numpy

How It Works:-
1. Login to the platform
2. Capture mood via webcam → CNN model classifies emotion
3. Recommend movies based on mood & liked preferences
4. Like movies → system refines future suggestions
5. Search & explore trending movies via TMDB API
