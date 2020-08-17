from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from sklearn.cluster import KMeans
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
print('loading tensorflow model')
model = hub.load(module_url)
print('finish loading model')
 # Name of the apps module package
app = Flask(__name__)
print('finish initialize flask')
# Load in the model at app startup
df = pd.read_csv('final_df.csv')
df_reviews = pd.read_json("reviews.json")
pipeline = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    user_name = features[0]
    user_query = features[1:]
    user_query = [int(i) for i in user_query]
    imdb_rating = user_query[5]
    user_query = user_query[:5]
    query = np.array(user_query).reshape(1, -1)
    query_scaled = pipeline['scaler'].transform(query.reshape(1, -1))
    _, ind = pipeline['knn'].kneighbors(query_scaled)
    df_rec = df.iloc[np.array(ind[0]),:]
    df_filtered = df_rec[df_rec.users_rating >= imdb_rating]        
    first_title = df_filtered.title_clean.values[0]
    second_title = df_filtered.title_clean.values[1]
    third_title = df_filtered.title_clean.values[2]
    first_img = df_filtered.img_url.values[0]
    second_img = df_filtered.img_url.values[1]
    third_img = df_filtered.img_url.values[2]
    first_rating = df_filtered.users_rating.values[0]
    second_rating = df_filtered.users_rating.values[1]
    third_rating = df_filtered.users_rating.values[2]
    first_desc = df_filtered.description.values[0]
    second_desc = df_filtered.description.values[1]
    third_desc = df_filtered.description.values[2]
    
    
    
    def generate_most_prominent_review(data, sample_title):
        reviews = data[data.title == sample_title]['reviewText']
        corpus = reviews.str.cat(sep=' ')
        sentence = nltk.sent_tokenize(corpus)
        df = pd.DataFrame(sentence, columns = ['sentence'])
        sentences = df['sentence'].copy()
        def embed(input):
            return model(input)
        # Vectorize sentences
        sentence_vectors = embed(sentences)
        k = 3
        # Instantiate the model
        kmeans_model = KMeans(n_clusters=k, random_state=42)
        # Fit the model
        kmeans_model.fit(sentence_vectors);
        top_sentences = []
        for i in range(k):
             #Define cluster centre
            centre = kmeans_model.cluster_centers_[i]
             #Calculate inner product of cluster centre and sentence vectors
            ips = np.inner(centre, sentence_vectors)
             #Find the sentence with the highest inner product
            top_index = pd.Series(ips).nlargest(1).index
            top_sentences.append(sentences[top_index].iloc[0])
        return top_sentences[0], top_sentences[1], top_sentences[2]
    
    first_review_1, first_review_2, first_review_3 = generate_most_prominent_review(data = df_reviews,
                                                                                    sample_title = first_title)
    second_review_1, second_review_2, second_review_3 = generate_most_prominent_review(data = df_reviews,
                                                                                    sample_title = second_title) 
    third_review_1, third_review_2, third_review_3 = generate_most_prominent_review(data = df_reviews,
                                                                                    sample_title = third_title) 
    return render_template('results.html', requirement =  f'Updating sample user requirement ...',
                           profound_score = f'Deep Profound Score = {user_query[0]}',
                           music_score = f'Entertaining Music Score = {user_query[1]}',
                           realistic_score = f'Realistic Settings Score = {user_query[2]}',
                           exciting_score = f'Experience Excitement Score = {user_query[3]}',
                           fun_score = f'Fun Score = {user_query[4]}',
                           imdb_score = f'IMDB Minimum Score = {imdb_rating}', 
                           opener = f'Movies recommended for {user_name}:',
                           prediction_title_1 = f'{first_title}',
                           prediction_title_2 = f'{second_title}',
                           prediction_title_3 = f'{third_title}',
                           prediction_img_1 = f'{first_img}',
                           prediction_img_2 = f'{second_img}',
                           prediction_img_3 = f'{third_img}',
                           prediction_rating_1 = f'{first_rating}',
                           prediction_rating_2 = f'{second_rating}',
                           prediction_rating_3 = f'{third_rating}',
                           prediction_desc_1 = f'{first_desc}',
                           prediction_desc_2 = f'{second_desc}',
                           prediction_desc_3 = f'{third_desc}',
                           prediction_t1_rev1 = f'{first_review_1}',
                           prediction_t1_rev2 = f'{first_review_2}',
                           prediction_t1_rev3 = f'{first_review_3}',
                           prediction_t2_rev1 = f'{second_review_1}',
                           prediction_t2_rev2 = f'{second_review_2}',
                           prediction_t2_rev3 = f'{second_review_3}',
                           prediction_t3_rev1 = f'{third_review_1}',
                           prediction_t3_rev2 = f'{third_review_2}',
                           prediction_t3_rev3 = f'{third_review_3}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True, port=5000)