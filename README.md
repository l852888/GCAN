# GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media
* Abstract:

This paper solves the fake news detection problem under a more realistic scenario. Given the source short-text tweet and its retweet users without text comments, we aim at predicting whether it is fake or not, and generating explanation by highlighting the evidences on suspicious retweeters and the words they concern. We develop a novel neural networkbased model, Graph-aware Co-Attention Networks (GCAN) to achieve the goal. Extensive experiments on real tweet datasets exhibit that GCAN can significantly outperform state-ofthe- art methods by 16% in accuracy on average,and produce reasonable explanation.

<img src="https://github.com/l852888/GCAN/blob/master/figure/model.PNG" width="75%" height="75%">
  
We develop a novel model, Graph-aware Co-Attention Networks (GCAN), to predict fake news based on the source tweet and its propagation based users. 

GCAN consists of five components. 
* The first is user characteristics extraction: creating features to quantify how a user participates in online social networking. 
* The second is new story encoding: generating the representation of words in the source tweet. 
* The third is user propagation representation: modeling and representing how the source tweet propagates by users using their extracted characteristics.
* The fourth is dual co-attention mechanisms: capturing the correlation between the source tweet and users’ interactions/propagation. 
* The last is making prediction: generating the detection outcome by concatenating all learned representations.


Datasets
------------------
Two well-known datasets compiled by Ma et al. (2017), Twitter15 and Twitter16, are utilized. Each dataset contains a collection of source tweets, along with their corresponding sequences of retweet users.

We choose only “true” and “fake” labels as the ground truth. Since the original data does not contain user profiles, we use user IDs to crawl user information via Twitter API.

Due to the privacy, we can not provide the data we crawled from the API.

Requirements
------------------
python >=3.5

Keras 2.2.4

scikit-learn 0.21

pandas 0.23.0

numpy 1.14.3

Citation
------------------------
Yi-Ju Lu and Cheng-Te Li. "GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media" The Annual Meeting of the Association for Computational Linguistics,ACL2020.

https://arxiv.org/abs/2004.11648?fbclid=IwAR2BCyJE4K4UqGY5l_f3VSj-VoFjidfWJPIG6SLQiPytNGdWhE1pdsnKmmM

