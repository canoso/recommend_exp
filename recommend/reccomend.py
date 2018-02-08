import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

u_cols = ['user_id', 'age', 'sex', 'ocupation', 'zip_code']
users = pd.read_csv('/Users/o6512532/azure/recommend/ml-100k/u.user', sep='|', names=u_cols)
print(users.head())

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('/Users/o6512532/azure/recommend/ml-100k/u.data', sep='\t', names=r_cols)
ratings['date'] = pd.to_datetime(ratings['unix_timestamp'], unit='s')
print(ratings.head())

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('/Users/o6512532/azure/recommend/ml-100k/u.item', sep='|', names=m_cols, usecols=range(5), encoding = "latin1")
print(movies.head())

movie_rating = pd.merge(movies, ratings)
lens = pd.merge(movie_rating, users)

print(lens.title.value_counts()[:25])

movie_stats = lens.groupby('title').agg({'rating': [np.size, np.mean]})

print(movie_stats.sort_values(by=[('rating', 'mean')], ascending=False).head())

atleast_100 = movie_stats['rating']['size'] >= 100
print(movie_stats[atleast_100].sort_values(by=[('rating', 'mean')], ascending=False)[:15])

user_stats = lens.groupby('user_id').agg({'rating': [np.size, np.mean]})
print(user_stats['rating'].describe())
