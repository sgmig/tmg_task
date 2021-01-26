#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This files contains functions and classes used for clustering. The first versions of these functions 
#were developped and are explained and tested in the notebook clustering_tests.ipynb


import os
import re

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


# In[ ]:


def get_stopwords(new_words = None, langs = None, html = True, file_types = True):
    """
    Compiles list of stopwords. 
    new_words: list of custom stopwords we want to include.
    langs: list of nltk languages to include. By default it takes all of them.
    html: bool. Include curated list of html tags thay may have gotten through scrapping.
    file_types: bool. Include som common file extensions.
    """
    
    stop_words = []
    
    if langs is None:
        langs = stopwords.fileids()


    for lan in stopwords.fileids():
        stop_words.extend(stopwords.words(lan))

    # Get a list of curated html tags from the filter_tags file.
    # The list contains all html tags, and the ones that shouldn't be included
    # as stopwords are preceded by the simbol '#'.

    with open('filter_tags', 'r') as tags:
        stop_words.extend([tag.strip() for tag in tags.readlines() if (not tag.startswith('#') and len(tag.strip())>=3)])

    # Let's add a list of file types.
    
    file_types = ['pdf', 'jpg', 'jpeg', 'png', 'gif', 'exe', 'js', 'zip', 'tar', 'gz', '7z', 'rar']

    stop_words.extend(file_types)

    # Remove repetitions

    stop_words = list(set(stop_words))
    
    return stop_words


# In[ ]:


# Function for preprocessing raw text.

def remove_numbers_lower(text):
    """
    Preprocessing function to be passed to the tokenizer. 
    It will remove numbers that appear by themselves of between '-'.
    This removes things such as telephone numbers. 
    It also lowecases the entirer string.
    
    text: string. Full text to be treated. 
    """
    
    processed = re.sub(r'\b\d+\b', '',text)
    processed = processed.lower()
    
    return processed


# In[ ]:


def plot_pc(vectors, components = [0,1], title = None, **kwargs):
    """
    Scatter plot of the principal components specified by the columns.
    vectors: array-like or sparse matrix of size (n_samples * n_components).
    components: list or tuple with the two directions to take as x,y.
    title: string. Plot title.
    kwargs for plt.scatter.
    """
    plt.figure(figsize=(8,8))
    
    col_x, col_y = components

    x = vectors[:,col_x]
    y = vectors[:,col_y]
    plt.scatter(x,y, **kwargs)
    
    if title:
        plt.title(title)
    
    plt.xlabel(f'Component {col_x}')
    plt.ylabel(f'Component {col_y}')

    plt.show()


# In[ ]:


class ClusterDecissionHelper():
    """
    Suggest and optimal number of clusters K for KMeans based on their scores.
    
    This class allows to perform KMeans with different numbers of clusters and compute 
    the silhouette score, inertia, or both. Repeats the process several times, and analize
    the scores for the optimal K. 
    """
    
    def __init__(self, max_clusters, min_clusters = 2,
                 repeat = 5, metric = 'euclidean', **kwargs ):
        
        """
        kwargs: pass arguments to sklearn.cluster.KMeans()
        min_clusters: int. Lower bound for the number of clusters. Default is 2, the minimum
                   allowed value. 
        max_clusters: int. Upper bound for the number of clusters.
                      Maximum allowed value is (number of vectors - 1)
        """
        
        self.repeat = repeat
        self.metric = metric
        
        self.KMeans_kwargs = kwargs
        
        self.cluster_numbers = list(range(min_clusters,max_clusters+1))
        
        self.sil_scores = None
        self.inertia_scores = None
        
        
    def compute_scores(self, dt_matrix,
                       sil_score = False, inertia = False):
    
        """
        Computes arrays of the desired score, of size self.repeat x len(self.cluster_numbers)
        It need to be told which method to use.


        dt_matrix: array-like or sparse matrix. Document term matrix with docs to be clustered.
        sil_score: bool. Compute Silhouette score. Default is False. 
        inertia: bool. Compute inertia. Default is False.    
        """

        repeat = self.repeat
        cluster_numbers = self.cluster_numbers
        
        doc_number = dt_matrix.shape[0]

        # Build array to keep the results of each run
        sil_array = np.zeros((repeat, len(cluster_numbers)))
        inertia_array = np.zeros((repeat, len(cluster_numbers)))

        for c_idx,c_number in enumerate(cluster_numbers):

            model = KMeans(n_clusters=c_number, **self.KMeans_kwargs)

            for repetition in range(repeat):
                # Fit the model and get the labels
                model.fit(dt_matrix)
                labels = model.labels_

                # Compute scores

                if sil_score:
                    sil_array[repetition, c_idx] = silhouette_score(dt_matrix, labels, metric = self.metric)
                if inertia:
                    inertia_array[repetition, c_idx] = model.inertia_

        if sil_score:
            self.sil_scores = sil_array 
        if inertia:
            self.inertia_scores = inertia_array
            
    def get_K_from_silhouettes(self):
        """ 
        Identifies optimal K from silhouette scores obtained with compute_scores()
        If two or more values of K are tied, the smallest one is selected.
        """
        
        scores = self.sil_scores

        # The -1 factor and kind='stable' ensure that if two or more values tie, the smaller one
        # is kept. 
        
        ranked_K_indices = np.argsort(-1*np.bincount(scores.argmax(axis=1)), kind='stable')

        winning_K = self.cluster_numbers[ranked_K_indices[0]]
        
        self.best_K_sil = winning_K

        return winning_K
    
    def plot_scores(self, method = 'sil'):
        """Convinience method for plotting scores obtained with compute_scores().
        method: string. 'sil' for silhouettes, or 'in' for inertia. 
        """

        titles = {'sil': 'Silhouette score', 'in': 'Inertia' }

        scores = {'sil': self.sil_scores, 'in':self.inertia_scores}
        
        score_arr = scores[method]
        title = titles[method]

        fig, ax = plt.subplots(figsize = (8,8))

        for i in range(self.repeat):
            ax.plot(self.cluster_numbers, score_arr[i,:])
            
        plt.xticks(self.cluster_numbers)
        plt.xlabel('Cluster Number')
        plt.ylabel('Score')
        plt.title(f'{title} for {self.repeat} repetitions.')

        plt.grid(True)

        plt.show()
        
    @staticmethod
    def elbow_finder(x,y, plot = False):
        """Finds an elbow in the graph of y=f(x) by finding the maximum distance between
        f(x) and the line that joins (x_0, y_0) with (x_final, y_final).
        Depends strongly on the noise.
        x,y: One dimensional arrays of values.
        
        NOTE: Plotting requires the input to be arrays. If no plot is required, we can pass lists. 
        """
        
        # initial point
        pi = np.array([x[0], y[0]])
        
        # final point
        
        pf = np.array([x[-1], y[-1]]) 
        
        # unit vector parallell to the line
        
        n_vec = (pf - pi) / np.linalg.norm((pf-pi))
        
        distances = np.zeros(len(x))
        
        # I don't need the inital and final point. Distance is 0 by definition.
        for i in range(len(x[1:])):
            
            q =np.array([x[i],y[i]])
            
            dif_vec= pi-q
            
            d = np.linalg.norm( (dif_vec) - np.dot(dif_vec,n_vec) * n_vec )
            
            distances[i] = d
            
            elbow_x_id = distances.argsort(kind='stable')[::-1][0]
            
            elbow_x = x[elbow_x_id]
            
        if plot:
            
            plt.figure(figsize=(8,8))
            plt.plot(x, y)
            plt.plot(x[[0,-1]], y[[0,-1]], c = 'g')
            plt.axvline(elbow_x, c='r')
            
        return elbow_x


# In[ ]:


def interpret_clusters(fitted_model, inv_vocabulary, reducer = None, n_words = 10):
    """
    Gets the first n most important words corresponding to each cluster centroid. 
    It needs the fitted KMeans object, and and inverse dictionary built from the vectorizer to look
    for words by id.     
    If dimensional reduction was performed, we also need the TSVD object used for dimensional reduction.
    
    Returns a pandas Dataframe with words and weights for each cluster.
    """
    inv_vocab = inv_vocabulary
    
    if reducer is None:
        centers = fitted_model.cluster_centers_
    else:
        centers = reducer.inverse_transform(fitted_model.cluster_centers_)
    
    #words = {}
    #weights = {}
    cluster_dict = {}
    for i in range(centers.shape[0]):
        vec = centers[i,:]
        argsorted = np.argsort(vec)[::-1]

        cluster_dict[f'Cluster_{i}'] = {'words': [inv_vocab[i] for i in argsorted[:n_words]], 
                                        'weights': vec[argsorted[:n_words]] }
        
        #words[f'Cluster_{i}'] = [inv_vocab[i] for i in argsorted[:n_words]]
        #weights[f'Cluster_{i}'] = vec[argsorted[:n_words]] 
        
        
    return pd.DataFrame(cluster_dict).transpose()

