
# Importing  liberaries 

from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Loading CSV file
movies = pd.read_csv(r'./movies.csv')


# Preprocessing

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['feature'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Fetching Posters

def fetch_poster(movie_id, retries=20):
    for attempt in range(retries):
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            return None
        except requests.exceptions.RequestException as e:
            print(f"Retry {attempt+1} failed for {movie_id}: {e}")
    return None

# Getting Recommendation

def get_recommendations(movie):
    try:
        index = movies[movies['title'] == movie].index[0]
    except IndexError:
        return [{
            'title': ' Movie not found',
            'poster': '../static/not-found.svg'
        }]  # movie not found
    
    distances = sorted(
        list(enumerate(cosine_sim[index])), 
        reverse=True, 
        key=lambda x: x[1]
    )

    recommendations = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]]['movie_id']
        title = (movies.iloc[i[0]]['title']).capitalize()
        poster = fetch_poster(movie_id)
        recommendations.append({
            'title': title,
            'poster': poster
        })

    return recommendations


# Creating routes for Commands

@app.route('/')
def home():
    return render_template("index.html", recommendations=None)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = (request.form['movie']).lower()
    recommendations = get_recommendations(movie_name)
    return render_template('index.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
