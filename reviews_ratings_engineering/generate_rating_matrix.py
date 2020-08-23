import pandas as pd
import numpy as np
import gzip
import os
import json

from generate_reviews_adjectives import DataLoad


data_dir = os.path.join('/Users/jiristodulka/GoogleDrive/GitHub/product_filter','data')
imdb_filters = pd.read_csv('imdb_amazon_movie_vectors.csv')




def utility_matrix_info(merged_df):
    dime = merged_df.reviewerName.nunique()*(merged_df.title_clean.nunique())
    count_ratings = merged_df.overall.count()
    print('There are {0}: userss'.format(merged_df.reviewerName.nunique()))
    print('There are {0}: movie titles'.format(merged_df.title_clean.nunique()))
    print('The utility matrix has: {0} entries'.format(dime))
    print('There are {0}: ratings in the utility matrix'.format(count_ratings))
    print('Sparsity: {0}'.format((1-(count_ratings/dime))*100),'%')




def subsample_utility_matrix(ratings, imdb_filters,  reviewer_rating_min, title_rating_min):

    ratings = ratings[ratings['reviewerName']!= 'Amazon Customer']
    ratings = pd.merge(ratings, imdb_filters, how='inner', left_on='title',
                      right_on='title_origin_amazon')[["reviewerName", "title_clean", "overall"]]

    ratings['reviewer_rating_count'] = ratings.groupby(
        'reviewerName')['overall'].transform('count')
    ratings['title_rating_count'] = ratings.groupby(
        'title_clean')['overall'].transform('count')
    subsample = ratings[(ratings['reviewer_rating_count'] > reviewer_rating_min) & (
        ratings['title_rating_count'] > title_rating_min)].drop(['reviewer_rating_count', 'title_rating_count'], axis=1)
    utility_matrix_info(subsample)
    return subsample



def main():
    dataload = DataLoad(data_dir)
    meta_df = dataload.load_meta()
    reviews_df = dataload.load_reviews()
    merged_df = dataload.merge_reviews_meta()
    ratings_app = subsample_utility_matrix(merged_df, imdb_filters, reviewer_rating_min = 50, title_rating_min = 150)
    return ratings_app

if __name__ == "__main__":
    ratings_app =  main()
    ratings_app.to_csv('ratings_app.csv', index = False)
