import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import MinMaxScaler

data_dir = '/Users/jiristodulka/GoogleDrive/GitHub/Netflix_Movie_Filter/data'
file_path = os.path.join(data_dir, "imdb_scraped_less.txt") #<- these are the scraped movie profiles from IMDb
sampled_reviews = pd.read_csv(data_dir + '/sampled_reviews.csv') #<- thes is the final deliverable of generate_reviews_adjectives.py

def clean_titles_IMDb():
    '''
    Desc.:
    Input:
    Output:
    '''

    df = pd.read_json(file_path, orient = 'records')
    votes = [s.replace(',', '') if s is not None else None for s in df.votes]
    df['votes_int'] = pd.to_numeric(votes, errors='coerce')
    df_filtered = df.drop_duplicates(['title'])
    df_filtered['title_clean'] =  df_filtered.title.str.replace('[^a-zA-Z0-9]',' ')
    df_filtered['title_clean'] =  df_filtered.title_clean.str.strip()
    df_filtered['title_clean'] =  df_filtered.title_clean.str.replace('VHS',' ')
    df_filtered['title_clean'] =  df_filtered.title_clean.str.replace('Blu ray',' ')
    df_filtered['title_clean'] =  df_filtered.title_clean.str.replace('   ',' ')
    df_filtered['title_clean'] =  df_filtered.title_clean.str.replace('  ',' ')
    return df_filtered

def clean_titles_sampled_reviews():
    '''
    Desc.:
    Input:
    Output:
    '''
    sampled_reviews['title_clean'] =  sampled_reviews.title.str.replace('[^a-zA-Z0-9]',' ')
    sampled_reviews['title_clean'] =  sampled_reviews.title_clean.str.strip()
    sampled_reviews['title_clean'] =  sampled_reviews.title_clean.str.replace('VHS',' ')
    sampled_reviews['title_clean'] =  sampled_reviews.title_clean.str.replace('Blu ray',' ')
    sampled_reviews['title_clean'] =  sampled_reviews.title_clean.str.replace('DVD',' ')
    sampled_reviews['title_clean'] =  sampled_reviews.title_clean.str.replace('   ',' ')
    sampled_reviews['title_clean'] =  sampled_reviews.title_clean.str.replace('  ',' ')
    sampled_reviews['title_clean'] =  sampled_reviews.title_clean.str.strip()
    return sampled_reviews


def common_titles_filters(df_filtered,sampled_reviews):
    '''
    Desc.:
    Input:
    Output:
    '''
    df = sampled_reviews[sampled_reviews['title_clean'].isin(df_filtered['title_clean'])] 
    df_filters = df.loc[:,['title', 'title_clean', 'review_adjectives']]
    return df_filters



def identify_paterns(df_filters):
    '''
    Desc.:
    Input:
    Output:
    '''

    dict_paterns = {'patern_memorable' : 'deep|memorable|happy|poetic|noble',
        'patern_good_music' : 'entertaining|musical|music|score',
        'patern_feel_at_the_scene' : 'believable|accurate|correct|likable',
        'pater_goosebump' : 'special|worth|sexual|mysterious|dangerous',
        'patern_heart_lifting_humor' : 'dumb|comic|funny|humorous|hilarious|clever|witty'}

    for key, patern in dict_paterns.items():
        df_filters['isin_{0}'.format(key)] = df_filters['review_adjectives'].str.contains(patern).astype(int)
    return df_filters

def aggregate_filters(df_filters, filters_list = ['memorable','good_music','feel_at_the_scene','goosebump','heart_lifting_humor']):
    '''
    Desc.:
    Input:
    Output:
    '''
    scaler = MinMaxScaler( feature_range=(0, 5))
    df_filters_aggregated = (df_filters.groupby(['title', 'title_clean']).sum()) * 10 
    filters_ = pd.DataFrame(scaler.fit_transform(df_filters_aggregated.loc[:,'isin_patern_memorable': ]), columns = filters_list).round()
    return df_filters_aggregated, filters_


def movie_profiles(df_filters_aggregated, filters_, df_filtered):
    '''
    Desc.:
    Input:
    Output:
    '''
    df_filters_agg_scaled = pd.concat([df_filters_aggregated.reset_index(), filters_], axis=1)
    imdb_amazon_movie_vectors = pd.merge(df_filtered, df_filters_agg_scaled, how = 'inner',
        left_on = 'title_clean', right_on = 'title_clean').drop_duplicates('title_clean')
    imdb_amazon_movie_vectors = imdb_amazon_movie_vectors.rename(columns={'title_x': 'title_origin_imdb',
        'title_y': 'title_origin_amazon'})
    return imdb_amazon_movie_vectors


def main():
    df_filtered = clean_titles_IMDb()
    sampled_reviews = clean_titles_sampled_reviews()
    df_filters = common_titles_filters(df_filtered,sampled_reviews)
    df_filters = identify_paterns(df_filters)
    df_filters_aggregated, filters_ = aggregate_filters(df_filters)
    imdb_amazon_movie_vectors = movie_profiles(df_filters_aggregated, filters_, df_filtered)

    return imdb_amazon_movie_vectors


if __name__ == "__main__":
    imdb_amazon_movie_vectors =  main()
    imdb_amazon_movie_vectors.to_csv('imdb_amazon_movie_vectors.csv', index = False)