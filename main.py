# Data
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Matrix
from scipy.sparse import csr_matrix 

# Model
from sklearn.neighbors import NearestNeighbors

# INTRODUCTION

# IMPORT
df_movies = pd.read_csv("movie.csv")
df_ratings = pd.read_csv("rating.csv", low_memory=False)

def genres_list(row):
  return str(row).split(sep='|')

df_movies['genres'] = df_movies['genres'].apply(genres_list)

df_ratings = df_ratings.dropna()

df_merged = df_movies.merge(df_ratings, on='movieId')

# EDA

df_avg = df_merged.groupby('title').agg({'rating' : ['count', 'mean']}) # EXPLORATION
df_avg.columns = ['count', 'mean']
df_avg = df_avg.sort_values(ascending=False, by=['count'])

# VISUALIZATIONS
plt.figure(figsize=(22, 6)) # PLOT 1
mean_plot = sns.histplot(data=df_avg, x=df_avg['mean'], bins=100)
mean_plot.set(xlabel='Mean', ylabel='Count', title='Plot 1: Count of Ratings vs. Average Score')
fig = mean_plot.get_figure()
fig.savefig("mean_plot.png")

plt.figure(figsize=(22, 6)) # PLOT 2
mean_plot = sns.countplot(data=df_avg, x=df_avg['mean'].apply(np.ceil))
mean_plot.bar_label(mean_plot.containers[0])
mean_plot.set(xlabel='Mean', ylabel='Count', title='Plot 2: Count of Ratings vs. Average Score (Ceiled)')
fig = mean_plot.get_figure()
fig.savefig("mean_plot_ceiled.png")

plt.figure(figsize=(18, 8)) # PLOT 3
count_plot = sns.histplot(data=df_avg, x='count', bins=50)
count_plot.set(xlabel='Count of Ratings', ylabel='Frequency', title='Plot 3: Frequency of Ratings vs. Count of Ratings')
fig = count_plot.get_figure()
fig.savefig("count_plot.png")

plt.figure(figsize=(21, 8)) # PLOT 4
merged_plot = sns.jointplot(data=df_avg, x='mean', y='count')
merged_plot.set_axis_labels('Count of Ratings', 'Mean of Ratings')
merged_plot.fig.suptitle('Plot 4: Count of Ratings vs. Mean of Ratings (Merged)')
plt.tight_layout()
fig = merged_plot.fig
fig.savefig("merged_plot.png")

# ANALYSIS
df_clean = df_avg.loc[df_avg['count'] >= 12]

df_merged = df_merged[df_merged['title'].isin(df_clean.index)]

df_movie_userid = df_merged.pivot_table(index="title", columns="userId", values='rating').fillna(0)

df_movie_matrix = csr_matrix(df_movie_userid.values)

# MODELING

# K-Nearest Neighbors

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(df_movie_matrix)

def knn_predict(query_list):
  for query in query_list:
    query_idx = movies.index(query)
    distances, indices = knn.kneighbors(df_movie_userid.iloc[query_idx, :].values.reshape(1, -1), n_neighbors=6)
    for i in range(0, len(distances.flatten())):
      if i == 0:
        print('Recommendations for {0}:\n'.format(df_movie_userid.index[query_idx]))
      else:
        print('{0}: {1}'.format(i, df_movie_userid.index[indices.flatten()[i]]))
    print('\n------\n')

movies = list(df_movie_userid.index)
user_input = str(input())
input_list = [x for x in df_movie_userid.index if user_input.lower() in x.lower()]
knn_predict(input_list)


