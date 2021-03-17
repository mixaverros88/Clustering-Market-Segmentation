
# Exploratory Alaysis

The USCensus19910raw data set, it is a multivariate dataset which consist of 124 feature, the 54 is categorical.
https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29

1. age
2. citizenship
3. class of worker
4. English proficiency
5. number of children
6. immigration year
7. multilingual
8. mode of transport
9. marital status
10. looking for work
11. birthplace
12. workplace
13. race
14. income
15. sex
16. education

## Data Description



| Dataset     | Instances   | Features|
| ------------|:-----------:| -------:|
| Original    | 2.458.285   | 125     |
| New         | centered    | 14      |


As we can observe the dataset is equally distributed among sex.

TODO: visualize income, age, sex distribution

![alt text](https://raw.githubusercontent.com/mixaverros88/Clustering-Market-Segmentation/main/images/sexdist.png?token=ACYAUFMLJRSCJQMYUHMCYZ3AKGR4G)

## Data preprocessing

Another observation from the graph concerns the domains of the two variables Age and Income. We understand that the domain for Age is from around 20 to 70, whereas for Income it is from around 40,000 to over 300,000. Which points to a vast difference between the range of these values. Therefore, we must incorporate an important step in our analysis, and we must first standardize our data. Standardization is an important part of data preprocessing, which is why we’ve devoted the entire next paragraph precisely to this topic.


## Principal Components Analysis
By reducing the number of features, we’re improving the performance of our algorithm. 
we need to decide how many features we’d like to keep based on the cumulative variance plot
TODO: add plot as image

## K-Means Clustering
TODO: add elbot plot as image

## Hierachical Clustering

## DBSCAN