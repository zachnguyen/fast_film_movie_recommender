# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:55:34 2020

@author: Zach Nguyen
"""

import numpy as np

def process_user_input(features):
    """ Take user feature and create correct form
    """
    
    # Get user name
    user_name = features[0]
    
    # Take mood filter values, convert to integer and reshape to array
    user_query = features[1:6]
    user_query = [int(i) for i in user_query]    
    
    # Get imdb ratings
    imdb_rating = features[6]
    imdb_rating = float(imdb_rating)
    
    return user_name, user_query, imdb_rating

def filter_by_mood(df, user_query, pipeline, k):
    """ Take movie information dataframe and filter by knn model in the pipeline object, filtering the df to k rows
    """
    # Set param to take the number of movies to filter by mood
    pipeline.set_params(knn__n_neighbors = k)
    
    # Scale the user input using Standard Scaler
    user_query = np.array(user_query).reshape(1, -1)
    query_scaled = pipeline['scaler'].transform(user_query.reshape(1, -1))
    
    # Get indices of rows filtered by knn
    _, ind = pipeline['knn'].kneighbors(query_scaled)
    
    # filter the dataframe by indices extracted
    df_filtered = df.iloc[np.array(ind[0]),:].title_clean
    
    return df_filtered

def get_results_dict(df_filtered, three_most_prominent, no_titles):
    """ Take filtered dataframe and reviews dictionary and get the information to display to users
    """
    
    # Initialize dictionary object
    results = {}
    
    # Get dictionary of results to display
    for title_index in range(no_titles):
        results[f'title_{title_index + 1}'] = {'title' : df_filtered.title_origin_amazon.values[title_index],
                                              'img_url' : df_filtered.img_url.values[title_index],
                                              'imdb_rating' : df_filtered.users_rating.values[title_index],
                                              'description': df_filtered.description.values[title_index],
                                              'reviews' : three_most_prominent[df_filtered.title_origin_amazon.values[title_index]]}
    return results