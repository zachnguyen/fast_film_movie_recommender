from flask import Flask, request, render_template
from predict_external_user import main as recommend 
from helper_function import process_user_input, filter_by_mood, get_results_dict

import json
import pandas as pd
import pickle

# Bellow import imports script and main() function `recommend' for recommendations upon the user's history

 # Name of the apps module package
app = Flask(__name__)
print('finish initialize flask')

# Load in the model at app startup
df = pd.read_csv('imbd_amazon_movie_vectors.csv')


# Load pipeline for mood filter
pipeline = pickle.load(open('finalized_model.sav', 'rb'))

# Load the review dictionary
with open('three_most_prominent_reviews.json') as f:
  three_most_prominent = json.load(f) 

# Set Home page
@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('index.html')

 
# Prediction endpoint
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """ Make recommendations of movies based on user input, returns movie list 
    """
    
    # Make movie recommendations based on user history file
    if request.method == 'POST':
        f = request.files['csvfile']
        history_df = pd.read_csv(f)
    recommendations = recommend(history_df)
    
    # Take user input
    features = [x for x in request.form.values()]
    # Extract username, mood queries and imdb requirement
    user_name, user_query, imdb_rating = process_user_input(features)
    # Filter dataframe by mood
    titles_order_filters = filter_by_mood(df = df,
                                         user_query = user_query,
                                         pipeline = pipeline,
                                         k = 4300)
    
    # Match mood recommendations with user profile
    df_filtered =  df.set_index('title_clean')
    df_filtered = df_filtered.loc[titles_order_filters, :].reset_index()
    df_filtered = df_filtered[df_filtered['title_clean'].isin(recommendations.title_clean)]
    
    # Filter dataframe by imdb
    df_filtered = df_filtered[df_filtered.users_rating >= imdb_rating]

    # Return results as dictionary to be passed to result page
    results = get_results_dict(df_filtered = df_filtered,
                               three_most_prominent = three_most_prominent,
                               no_titles = 10)
    
    return render_template('results.html', 
                           requirement =  'Updating sample user requirement ...',
                           user_input = user_query,
                           imdb_score = f'IMDB Minimum Score = {imdb_rating}',
                           opener = f'Displaying recommendations for {user_name} ...',
                           prediction = results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True, port=5000)