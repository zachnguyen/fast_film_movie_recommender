import pandas as pd
import numpy as np
import gzip
import os
import json
import matplotlib 
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
import string

import textblob
from textblob import TextBlob


data_dir = os.path.join('/Users/jiristodulka/GoogleDrive/GitHub/product_filter','data')


class DataLoad:
    '''
    Desc.
    '''    
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_meta(self):
        '''
        Desc.: 
            - loads metadata that sore info about 'Movies & TV' ONLY
        Input:
            - By Default it takes the data from ./data directory
        Returns:
            - meta_df : pd.DataFrame  object with info about items
            
        '''
        meta = []
        with gzip.open(self.data_dir  +'/meta_Movies_and_TV.json.gz') as f:
            for l in f:
                meta.append(json.loads(l.strip()))
                
        self.meta_df = pd.DataFrame(meta)[['main_cat', 'title','asin']]
        self.reviews_df =self. meta_df[self.meta_df['main_cat']== 'Movies & TV']
        return  self.meta_df


    def load_reviews(self):
        '''
        Desc.:
            - Load Reviews
        Input:
            - By Default it takes the data from ./data directory
        Returns:
            reviews_df: pd.DataFrame object storing ALL the reviews for MULTIPLE CATEGORIES
            
        '''
        reviews = []
        for line in open(self.data_dir  + '/Movies_and_TV_5.json', 'r'):
            reviews.append(json.loads(line))
        
        self.reviews_df = pd.DataFrame(reviews)
        return self.reviews_df

    def merge_reviews_meta(self):
        '''
        
        '''
        self.merged_df = pd.merge(self.reviews_df, self.meta_df[['title', 'asin']],
                            how = 'inner', left_on='asin', right_on = 'asin')
        self.merged_df['char_count'] = self.merged_df['reviewText'].str.len()
        return self.merged_df


def downsample_reviews(merged_df, rating_min = 10 ,length = [300,800]):
    '''
    Desc.: 
        - Subsets the merged_df input to extract only relevant records ("Movies and TV"):
            1. selects only movies category
            2. N/A
            3. length of reviews in certain range
            4. only certain # of reviews
    Input:
       - merged_df: output of merge_reviews_meta(reviews_df, meta_df)
       - length: min and max length of reviews in range
       - trashold: max number of reviews per movie
       
     Returns:
         downsampled reviews pd.DataFrame
    '''
    down_reviews_df = merged_df.copy()
    
    down_reviews_df['char_count'] = down_reviews_df['reviewText'].str.len()
    down_reviews_df['sum_reviews'] = down_reviews_df.groupby('title')['title'].transform('count')

    sample = down_reviews_df[down_reviews_df['char_count'].between(length[0], length[1])]
    sample =  sample[sample['sum_reviews'] >= rating_min]
    
    titles_index = sample.title.value_counts()[sample.title.value_counts()>=rating_min].index 
    sample = sample[sample['title'].isin(titles_index)]
    
    sample_df = sample.groupby('title').apply(lambda x: x.sample(rating_min)).reset_index(drop = True)

    return sample_df


def clean_reviews(sample_df):
    '''
    Desc.:
        Clean 'reviewText', extracts adjectives for each review into a list in new column: review_adjectives
    Input:
        - sample_df: pd.DataFrame as sampled reviews
    Output:
        - ...: identical to the input but with new columns storing the adjectives in review's in the  a list
    
    '''
    clean_sample = sample_df.copy()
    clean_sample['reviewText']=clean_sample.reviewText.str.lower()
    clean_sample['reviewText'] = clean_sample['reviewText'].str.replace('[^A-z ]','').str.replace(' +',' ').str.strip()
    
    def get_adjectives(text):
        blob = TextBlob(text)
        '''
        Extracts adjectives
        '''
        return [ word for (word,tag) in blob.tags if tag == "JJ"]
    
    clean_sample_df = clean_sample.copy()
    clean_sample['review_adjectives'] = clean_sample['reviewText'].apply(get_adjectives)
    clean_sample_df = clean_sample.copy()
    return clean_sample_df




def main():
    dataload = DataLoad(data_dir)

    meta_df = dataload.load_meta()
    reviews_df = dataload.load_reviews()
    merged_df = dataload.merge_reviews_meta()

    sample_df = downsample_reviews(merged_df)
    clean_sample_df =  clean_reviews(sample_df)
    return clean_sample_df

if __name__ == "__main__":
    clean_sample_df =  main()
    clean_sample_df.to_csv('sampled_reviews.csv', index = False)