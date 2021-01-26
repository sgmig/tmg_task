## Website clustering task

The goal of this repository is to use KMeans clustering algorithm in order to find clusters among a list of websites. 

* Given a file with a list of url's, scrape text from each site's homepage, and from all the links that appear in the homepage.

* Using the scrapped text as features of each site, use KMeans to find clusters among them. 

## Directories

#### sites_lists
stores lists of sites to scrape.

#### sites_contents
stores the text files that result from scrapping sites.

#### logs
reports of scrapping.

#### new_contents
store contents of sites that were not used for training.

#### cluster_results
list of sites and their cluster labels.
list of common words and weights for each cluster.
pickled models, vectorizer and TSVD objects (for dimensional reduction).


## Files detail

#### scraptools.py
Scrapping Tools: Functions and classes used for scrapping. The most important method is scrape_full, which takes a list of URLs, scrapes the main site and the links for text, and writes each site's content to a text file. It also produces a log for the scrapping process, and a links report for the sit if too many of it's links fail.

#### clustools.py
Clustering tools. Functions are classes for text preprocessing and clustering. It includes methods for defining a list of stop words and a preprocessor to be passed to the tokenizer, a shorthand method to plot data using it's principal componenents, and a method to easily get the most prevalent words for each cluster. 

The most important tool in this file is the class ClusterDecissionHelper. This class allows to perform KMeans with different numbers of clusters and compute 
the silhouette score, inertia, or both. It can repeat the process several times, and analize the scores for the optimal suggesting and optimal value of K for KMeans. It also provides a rudimentary static method to automatically find elbows in graphs like inertia vs. number of clusters or for the plot of the singular values of each dimension obtain through SVD, based on maximizng the distances between the plotted curve and a straight line that goes through its ends.

#### scrapping.ipynb
Performs scrapping.
This file was originally used for testing the scrapping functions. The functions in scraptools.py wer first developped here and then mover over. 
The original function definitions were replaced by an import. 

#### fit_clustering.ipynb
This notebook takes textfiles corresponding to each site, and performs the following task
* Eliminate texts that are too short or empty.
* Perform Tf-Idf vectorizarion using a custom preprocessor and stopwords, generating a document term matrix
* Perform LSA (using Trucated singular value decomposition TSVD) keeping a large number of dimensions, and suggests the optimal amount of dimensions to keep by using the elbow method on the plot of the singular values.
* Reduce dimensionality of the document term matrix using the number of dimensions obtained in the previous step.
* Use clustools.ClusterDecissionHelper to find the best value for the number of clusters, based on silhouette scores. 
* Perform KMeans clustering on the dimensionally reduced data.
* Assign cluster labels to each text (i.e. site) and save the results to a csv file.
* Get the most representative words of each cluster and save the results to a csv file.
* Save the fitted KMeans model, the vectorizer and the object (used to perform dimensional reduction) in pickle format. 

#### find_cluster_labels
The idea of this notebook is to illustrate what the classification process would look like once we have a trained model. 
The fisrt part of this notebook takes a csv file with a list of sites, and performs scrapping with the methods of scraptools.py.
The second part takes the texts generated, cleans them, and then loads the vectorizer and the TSVD obejct that were used in processing the data for fitting the clustering model. It transform the new data using these methods, and then loads the KMeans model  and assings them a cluster label. It then saves a csv with the list of sites and their corresponding clusters. For improvement: Make a pipeline for treating the data.  

#### clustering_tests.ipynb

This file was used for testing the data treatment and clustering process. Overall it performs the same tasks as fit_clustering.ipynb (except saving results and models), but it contains comments, plots, and other visualizations along the process. The functions of clustools.py were first developped in this notebook and then copied over. Function definitions in this notebook were replaced by imports.
This notebook is mainly useful for demonstration and discussion, and for preliminary analysis in order to choose hyperparameters. 
For carrying out the training process once parameters have been chosen it is recomended to use the fit_clustering notebook instead.

clusterting_test.ipynb could be expanded to include the classifications of new data once the model has been trained, and include analysis of the results. 



