# Hashtag-based-topic-modeling
This repository introduces a hybrid topic modeling approach for short tagged social media texts such as tweets. 

Conventional topic models like Latent Dirichlet Allocation (LDA) and its variants widely used to automatically extract thematic information from regular-sized documents fail to discover essential information from short texts. Short text documents such as tweets, comparing to regular-sized documents such as news articles, lack word co-occurrence information, which leads to very sparse and high dimensional vector representations. This extreme sparsity brings challenges to applying the conventional topic models on short texts. To address this sparseness problem, we compare the following four topic models:

1. Apply LDA directly on all tweets.

2. Apply LDA directly on tweets with at least one hashtag.

3. Apply LDA on aggregated tweets for each hashtag. In this approach, LDA is applied to text documents created for each hashtag by aggregating their tweet texts. 

4. Apply LDA on aggregated tweets for each cluster of relevant hashtags, which is our proposed Hashtag-Cluster based Aggregation (HCA) technique. 

Models 1 to 3 treat the tweets as flat texts by ignoring any relationship among the hashtags. However, model 4 treats tweets as semi structured texts and includes extra information in the topic modeling of the tweets.
