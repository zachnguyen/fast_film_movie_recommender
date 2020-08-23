#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:22:54 2020

@author: jiristodulka
"""


import pandas as pd
import numpy as np
from surprise.reader import Reader
from surprise import dump

filters = pd.read_csv('imbd_amazon_movie_vectors.csv')
movies_repository_df = filters[['title_clean']]
ratings_app = pd.read_csv('ratings_app.csv')
#history = pd.read_csv('NetflixViewingHistory.csv')


def clean_titles_external_history(history):
    '''
    Desc.:
    Input:
    Output:
    '''
    external_history_df = history.copy()
    # cleaning title
    external_history_df['title_clean'] = external_history_df.Title.str.replace(
        '[^a-zA-Z0-9]', ' ').replace(regex=r'\Season.*$', value='').replace('   ', ' ').replace(
        '  ', ' ').str.strip(
    )

    #variable: seen
    external_history_df['overall'] = 1

    # clening dataframe
    external_history_df.drop_duplicates(subset=['title_clean'], inplace=True)
    external_history_df.drop('Date', axis=1, inplace=True)
    print('Titles in the History: {0}'.format(
        external_history_df['title_clean'].nunique()))
    return external_history_df

def construct_common_prediction(movies_repository, external_history, user_name = 'appuser'):
    '''
    Desc. Makes two DFs: oNe with matching titles in our repository the user watched on Netflix
            and one with titles for prediction
    Input:
    Output:
    '''

    common_titles_df = pd.merge(movies_repository_df, external_history,
                                how='inner', left_on='title_clean', right_on='title_clean')
    common_titles_df['reviewerName'] = user_name
    common_titles_df = common_titles_df.loc[:, [
        "reviewerName", "title_clean", "overall"]]
    print('Titles matching our repository: {0}'.format(
        common_titles_df['title_clean'].nunique()))

    titles_for_prections_df = movies_repository_df[~movies_repository_df['title_clean'].isin(
        common_titles_df['title_clean'])]
    titles_for_prections_df['overall'] = 0
    titles_for_prections_df['reviewerName'] = user_name
    titles_for_prections_df = titles_for_prections_df.loc[:, [
        "reviewerName", "title_clean", "overall"]]
    print('Titles to be predictedy: {0}'.format(
        titles_for_prections_df['title_clean'].nunique()))

    return common_titles_df, titles_for_prections_df


def melt_ratings(ratings):
    """
    Desc.: Makes all combinations between users and titles with values 0 if nor rated (no roview - 
    we assume the user'didnt watch), 1 if review so user watched th movie
    """
    
    ratings = ratings[ratings['title_clean'].isin(movies_repository_df['title_clean'].unique())]
    ratings.drop_duplicates(subset=['reviewerName', 'title_clean'], inplace = True)
    # IF DOING PROBABILISTIC 
    ratings['overall'] = 1
    
    rating_pivot = pd.pivot_table(
    ratings, values='overall', index='reviewerName', columns='title_clean',  fill_value=0)
    rating_melt = pd.melt(rating_pivot.reset_index(), value_vars=rating_pivot.columns,
                      id_vars='reviewerName', value_name='overall')
    
    
    return rating_melt


def get_and_convert_ids(ratings):
    '''
    
    
    
    '''
    
    reviewer_id = ratings.loc[:,['reviewerName']]
    reviewer_id['reviewer_id'] = reviewer_id['reviewerName'].astype('category').cat.codes
    
    title_id = ratings.loc[:,['title_clean']]
    title_id['title_id'] =title_id['title_clean'].astype('category').astype('category').cat.codes

    ratings_ids = ratings.copy()
    ratings_ids['reviewer_id'] = reviewer_id['reviewer_id']
    ratings_ids['title_id'] = title_id['title_id'] 
    ratings_ids = ratings_ids.loc[:,['reviewer_id','title_id','overall' ]]
    
    
    reviewer_id.drop_duplicates(inplace = True)
    title_id.drop_duplicates(inplace = True)
    
    return reviewer_id, title_id, ratings_ids

def ids_user_watched(common_titles_df, title_id):

    titles_name_id = pd.merge(common_titles_df, title_id,
                              how='inner', left_on='title_clean', right_on='title_clean')
    titles_name_id.drop('overall', axis = 1, inplace = True)
    return titles_name_id



def load_models():
    _, algo_cosine = dump.load('algo_cosine_trained')
    _, algo_msd = dump.load('algo_msd_trained')
    _, algo_pearson = dump.load('algo_pearson_trained')
    return algo_cosine, algo_msd, algo_pearson


def predict(ids_user_watched_df, algo_cosine, algo_msd, algo_pearson, k):
    '''
    
    '''
    ids_user_watched_df['knn_cosine'] = ids_user_watched_df['title_id'].apply(lambda x: algo_cosine.get_neighbors(x, k))
    ids_user_watched_df['knn_msd'] = ids_user_watched_df['title_id'].apply(lambda x: algo_msd.get_neighbors(x, k))
    ids_user_watched_df['knn_pearson'] = ids_user_watched_df['title_id'].apply( lambda x: algo_pearson.get_neighbors(x, k))
    
    cosine_exploded = ids_user_watched_df.loc[:, ['reviewerName', 'title_clean', 'title_id', 'knn_cosine']].explode('knn_cosine')
    msd_exploded = ids_user_watched_df.loc[:, [ 'reviewerName', 'title_clean', 'title_id', 'knn_msd']].explode('knn_msd')
    pearson_exploded = ids_user_watched_df.loc[:, ['reviewerName', 'title_clean', 'title_id', 'knn_pearson']].explode('knn_pearson')

    
    exploded_df =  pd.concat([cosine_exploded, msd_exploded,pearson_exploded], axis=1, join='inner')
    exploded_df = exploded_df.loc[:,~exploded_df.columns.duplicated()]
    return exploded_df



def similarity_dataframes(exploded_df, title_id):
    '''


    '''
    cosine_df = pd.DataFrame(exploded_df['knn_cosine'].value_counts(normalize=True)).reset_index().rename(columns={'index': 'title_id',
                                      'knn_cosine': 'cosine_freq'})    
        
    msd_df = pd.DataFrame(exploded_df['knn_msd'].value_counts(normalize = True)).reset_index().rename(columns={'index': 'title_id',
                                  'knn_msd': 'msd_freq'})
    
    pearson_df = pd.DataFrame(exploded_df['knn_pearson'].value_counts(normalize = True)).reset_index().rename(columns={'index': 'title_id',
                                      'knn_pearson': 'pearson_freq'})
        
    similarities_df = pd.merge(cosine_df, msd_df).merge(pearson_df)
    similarities_df['mean_sim'] = similarities_df.loc[:,['cosine_freq', 'msd_freq', 'pearson_freq']].mean(axis = 1)
    similarities_df.sort_values('mean_sim',ascending=False, inplace = True)
    recommended_titles = pd.merge(similarities_df, title_id)
    return recommended_titles


def main(history):
    external_history_df = clean_titles_external_history(history)
    common_titles_df, _  = construct_common_prediction(movies_repository_df, external_history_df)
    ratings = melt_ratings(ratings = ratings_app)
    reviewer_id, title_id, ratings_ids = get_and_convert_ids(ratings)
    ids_user_watched_df = ids_user_watched(common_titles_df, title_id)
    algo_cosine, algo_msd, algo_pearson = load_models()
    exploded_df = predict(ids_user_watched_df,algo_cosine, algo_msd, algo_pearson, k = 20,)
    recommended_titles = similarity_dataframes(exploded_df, title_id)
    return recommended_titles    


    
    
    
    
    
    
    
    
    
    
    