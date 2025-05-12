from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile

app = Flask(__name__)
CORS(app)

# Unzip and load MovieLens dataset
def load_data():
    if not os.path.exists('ml-latest-small'):
        with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    return movies, ratings

movies, ratings = load_data()

# Preprocess genres for content-based filtering
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ')
tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Train SVD model for collaborative filtering
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd = SVD(n_factors=100, random_state=42)
svd.fit(trainset)

# Helper function to get hybrid recommendations
def get_hybrid_recommendations(user_ratings, n=10):
    # Get user-rated movie IDs
    rated_movies = [movie_id for movie_id, _ in user_ratings]
    
    # Collaborative filtering: Predict ratings for all movies
    predictions = []
    for movie_id in movies['movieId']:
        if movie_id not in rated_movies:
            pred = svd.predict(1, movie_id).est
            predictions.append((movie_id, pred))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_cf = [movie_id for movie_id, _ in predictions[:n*2]]
    
    # Content-based: Boost recommendations with genre similarity
    if rated_movies:
        sim_scores = np.zeros(len(movies))
        for movie_id in rated_movies:
            idx = movies[movies['movieId'] == movie_id].index[0]
            sim_scores += cosine_sim[idx]
        sim_scores[rated_movies] = 0  # Exclude already rated movies
        top_cb_idx = np.argsort(sim_scores)[-n*2:][::-1]
        top_cb = movies.iloc[top_cb_idx]['movieId'].tolist()
        
        # Combine CF and CB results (50-50 weight)
        combined = list(set(top_cf + top_cb))[:n]
    else:
        combined = top_cf[:n]
    
    # Return movie details
    recs = movies[movies['movieId'].isin(combined)][['movieId', 'title', 'genres']].to_dict('records')
    return recs

# API endpoints
@app.route('/api/movies', methods=['GET'])
def search_movies():
    query = request.args.get('query', '').lower()
    if query:
        results = movies[movies['title'].str.lower().str.contains(query)][['movieId', 'title', 'genres']].to_dict('records')
    else:
        results = movies[['movieId', 'title', 'genres']].sample(10).to_dict('records')
    return jsonify(results)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_ratings = data.get('ratings', [])  # List of [movieId, rating]
    recommendations = get_hybrid_recommendations(user_ratings)
    return jsonify(recommendations)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)