# GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media
![GCAN](https://github.com/l852888/GCAN/blob/master/figure/model.PNG)
Reference
------------------

Datasets
------------------
Two well-known datasets compiled by Ma et al. (2017), Twitter15 and Twitter16, are utilized. Each dataset contains a collection of source tweets, along with their corresponding sequences of retweet users. We choose only “true” and “fake” labels as the ground truth. Since the original data does not contain user profiles, we use user IDs to crawl user information via Twitter API.

Due to the privacy, we can not provide the data we crawled from the API.

Requirements
------------------
python >=3.5

Keras

scklearn,pandas,numpy

Model
-----------------
provide the code of GCN and dual co-attention layer 

Training
---------------------
provide the complete GCAN training

configuration file shows the hyperparameter setting of GCAN
