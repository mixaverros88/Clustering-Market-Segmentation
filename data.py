########################################################################## START ##########################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import time
import os

elapsed_time = {
    "kmeans": [],
    "gmm": [] ,
    "hierarchy": [],
    "dbscan": [] 
}

# dAncstry1,dAncstry2,iDisabl1,iDisabl2,dHour89,dHours,dIncome1,dIncome2,dIncome3,dIncome4,dIncome5,dIncome6,dIncome7,dIncome8,dIndustry,dOccup,iOthrserv,dPOB,iRelat1,iRelat2,iRiders,iRlabor,iRownchld,dRpincome,iRrelchldiSubfam1,iSubfam2,iTmpabsnt,dWeek89,iWork89,iWorklwk,iYearwrk

missing_values = ['n/a', 'na', '--', '?'] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
df=pd.read_csv(r''+my_path+'\\data\\USCensus1990.data.txt', sep=',', nrows=2000, na_values=missing_values)
print(df.shape)

# Data Preprocessing 

# iMilitary
# 0: No Military service
# 1: Military Service
df.loc[(df['iMilitary'] == 0) | (df['iMilitary'] == 4) , 'iMilitary'] = 0
df.loc[(df['iMilitary'] >= 1) & (df['iMilitary'] <= 3) , 'iMilitary'] = 1

groupedByUserId  = df.groupby(['caseid']) # group rows per user_id
df.insert(0, 'military', 0 , True)
for index,group in groupedByUserId:
    if group['iFeb55'].values[0] == 1 or group['iKorean'].values[0] == 1 or group['iMay75880'].values[0] == 1 or group['iRvetserv'].values[0] == 1 or group['iSept80'].values[0] == 1 or group['iVietnam'].values[0] == 1 or group['iWWII'].values[0] == 1 :
        df.loc[ df['caseid']== index, 'military'] = 1
    else:
        df.loc[ df['caseid']== index, 'military'] = 0

print(df)
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

# Start - Drop War columns 
df.drop('iFeb55', axis=1, inplace=True) 
df.drop('iKorean', axis=1, inplace=True)
df.drop('iMay75880', axis=1, inplace=True) 
df.drop('iRvetserv', axis=1, inplace=True)
df.drop('iSept80', axis=1, inplace=True)
df.drop('iVietnam', axis=1, inplace=True)
df.drop('iWWII', axis=1, inplace=True)
# End - Drop War columns 

df.drop('dYrsserv', axis=1, inplace=True) # year of active duty
df.drop('iMilitary', axis=1, inplace=True) # Military Srvc.

df.drop('caseid', axis=1, inplace=True)

df.drop('dHispanic', axis=1, inplace=True) # since tha most cases are not hispanic
df.drop(index=df[df['dPoverty'] == 0].index,    inplace=True) # drop the rows that has N/A
df.drop('iRemplpar', axis=1, inplace=True) # since iRemplpar column has many zero values
df.drop(index=df[df['iSchool']  == 0].index,     inplace=True)   # N/a Less Than 3 Yrs. Old
df.drop(index=df[df['iEnglish'] == 0].index,    inplace=True)    # N/a Less Than 5 Yrs. Old/speaks Only Eng
df.drop(index=df[df['iImmigr'] == 0].index,    inplace=True)     # Born in the U.S. since we can take this value from the citizen column

#TODO
#   dDepart 
#   dYrsserv vs iMilitary
#   iRiders
#   YEARWRK ==> Never Worked
#TODO

# iLang1
# 0: No 
# 1: Yes
df.loc[(df['iLang1'] == 0) | (df['iLang1'] == 2) , 'iLang1'] = 0

# iMobility
# 0: No 
# 1: Yes
df.loc[(df['iMobility'] == 0) | (df['iMobility'] == 2) , 'iMobility'] = 0

# iMobillim
# 0: No 
# 1: Yes
df.loc[(df['iMobillim'] == 0) | (df['iMobillim'] == 2) , 'iMobillim'] = 0

# iFertil
# 0: No 
# 1: 2-4
# 2: having many children 5-13
df.loc[(df['iFertil'] == 0) | (df['iFertil'] == 1) , 'iFertil'] = 0
df.loc[(df['iFertil'] >= 2) & (df['iFertil'] <= 4) , 'iFertil'] = 1
df.loc[(df['iFertil'] >= 5) & (df['iFertil'] <= 13) ,'iFertil'] = 2

# iRspouse
# 0: No 
# 1: Yes
df.loc[(df['iRspouse'] == 0) | (df['iRspouse'] == 6) , 'iRspouse'] = 0
df.loc[(df['iRspouse'] >= 1) & (df['iRspouse'] <= 5) , 'iRspouse'] = 1

# iPerscare
# 0: No 
# 1: Yes
df.loc[df['iPerscare'] == 2, 'iPerscare'] = 0

# dRearning
# 0: No 
# 1: Medium Earning
# 2: Rich
df.loc[(df['dRearning'] >= 1) & (df['dRearning'] <= 3) , 'dRearning'] = 1
df.loc[(df['dRearning'] >= 4) & (df['dRearning'] <= 5) , 'dRearning'] = 2

# dPwgt1
# 0: Slim 
# 1: Normal
# 2: Obese
df.loc[(df['dPwgt1'] == 2) | (df['dPwgt1'] == 3) , 'dPwgt1'] = 2

# iMeans
# 0: Not
# 1: Public Transportation
# 2: By own
# 3: Other
df.loc[(df['iMeans'] >= 2) | (df['iMeans'] <= 6) , 'iMeans'] = 1
df.loc[(df['iMeans'] == 1) | (df['iMeans'] >= 7) | (df['iMeans'] <= 10) , 'iMeans'] = 2
df.loc[ df['iMeans'] == 11, 'iMeans'] = 3
df.loc[ df['iMeans'] == 12, 'iMeans'] = 4

# iLooking
# 0: No 
# 1: Yes
df.loc[(df['iLooking'] == 0) | (df['iLooking'] == 2) , 'iLooking'] = 0
df.loc[ df['iLooking'] == 1, 'iLooking'] = 1

# iClass
# 0: No 
# 1: Yes
df.loc[(df['iClass'] == 0) | (df['iClass'] == 9) , 'iClass'] = 0
df.loc[(df['iClass'] != 0) & (df['iClass'] != 9) , 'iClass'] = 1

# iAvail
# 0: No 
# 1: Yes
df.loc[(df['iAvail'] >= 0) & (df['iAvail'] <= 3) , 'iAvail'] = 0
df.loc[ df['iAvail'] == 4, 'iAvail'] = 1

# iSchool
# 0: Not attend
# 1: Attend
df.loc[ df['iSchool'] == 1, 'iSchool'] = 0
df.loc[(df['iSchool'] >= 2) & (df['iSchool'] <= 3) , 'iSchool'] = 1

# iImmigr
# 0: Came to US before 1950
# 1: Came to US after 1950
df.loc[(df['iImmigr'] >= 1) & (df['iImmigr'] <= 9) , 'iImmigr'] = 0
df.loc[df['iImmigr'] == 10, 'iImmigr'] = 1

# iMarital
# 0: Never Married
# 1: Married 
df.loc[(df['iMarital'] >= 0) & (df['iMarital'] <= 3) , 'iMarital'] = 1
df.loc[df['iMarital'] == 4, 'iMarital'] = 0

# iYearsch
# 0: No School Completed
# 1: Median Education
# 3: High Education
df.loc[(df['iYearsch'] == 0) | (df['iYearsch'] == 1) , 'iYearsch'] = 0
df.loc[(df['iYearsch'] >= 2) & (df['iYearsch'] <= 10) , 'iYearsch'] = 1
df.loc[(df['iYearsch'] >= 10) & (df['iYearsch'] <= 17) , 'iYearsch'] = 2

# iEnglish      
# 0: Not Speak English
# 1: Speak English
df.loc[(df['iEnglish'] == 4) , 'iEnglish'] = 0
df.loc[(df['iEnglish'] >= 1) & (df['iEnglish'] <= 3) , 'iEnglish'] = 1

# iRagechld
# 0: No 
# 1: Yes
df.loc[(df['iRagechld'] == 0) | (df['iRagechld'] == 4) , 'iRagechld'] = 0
df.loc[(df['iRagechld'] >= 1) & (df['iRagechld'] <= 3) , 'iRagechld'] = 1

# dTravtime
# 0: No 
# 1: Below 1 hour
# 2: Above 1 hour
df.loc[(df['dTravtime'] >= 1) & (df['dTravtime'] <= 5) , 'dTravtime'] = 1
df.loc[(df['dTravtime'] == 6), 'dTravtime'] = 2

# iCitizen
# 0: Born in US
# 1: Not Born in US
df.loc[(df['iCitizen'] == 4) , 'iCitizen'] = 0
df.loc[(df['iCitizen'] >= 1) & (df['iCitizen'] <= 3) , 'iCitizen'] = 1

# iRPOB
# 0: Citizen
# 1: Not a Citizen
df.loc[df['iRPOB'] != 52, 'iRPOB'] = 0
df.loc[df['iRPOB'] == 52, 'iRPOB'] = 1

#Transform Categorical Data
df = pd.get_dummies(df, columns=['iRPOB'], prefix=['RPOB_Type_is'] )
df = pd.get_dummies(df, columns=['dIncome1'], prefix=['dIncome1_Type_is'] )
df = pd.get_dummies(df, columns=['dAge'], prefix=['dAge_Type_is'] )
df = pd.get_dummies(df, columns=['iYearsch'], prefix=['iYearsch_Type_is'] )
df = pd.get_dummies(df, columns=['iSchool'], prefix=['iSchool_Type_is'] )
df = pd.get_dummies(df, columns=['iSex'], prefix=['iSex_Type_is'] )
df = pd.get_dummies(df, columns=['iMarital'], prefix=['iMarital_Type_is'] )
df = pd.get_dummies(df, columns=['iImmigr'], prefix=['iImmigr_Type_is'] )
df = pd.get_dummies(df, columns=['iEnglish'], prefix=['iEnglish_Type_is'] )


df = pd.get_dummies(df, columns=['iLooking'], prefix=['iLooking_Type_is'] )
df = pd.get_dummies(df, columns=['iClass'], prefix=['iClass_Type_is'] )
df = pd.get_dummies(df, columns=['iAvail'], prefix=['iAvail_Type_is'] )
df = pd.get_dummies(df, columns=['military'], prefix=['military_Type_is'] )
df = pd.get_dummies(df, columns=['iLang1'], prefix=['iLang1_Type_is'] )
df = pd.get_dummies(df, columns=['iMobility'], prefix=['iMobility_Type_is'] )
df = pd.get_dummies(df, columns=['iFertil'], prefix=['iFertil_Type_is'] )
df = pd.get_dummies(df, columns=['iRspouse'], prefix=['iRspouse_Type_is'] )
df = pd.get_dummies(df, columns=['iPerscare'], prefix=['iPerscare_Type_is'] )
df = pd.get_dummies(df, columns=['dRearning'], prefix=['dRearning_Type_is'] )
df = pd.get_dummies(df, columns=['dPwgt1'], prefix=['dPwgt1_Type_is'] )
df = pd.get_dummies(df, columns=['iMeans'], prefix=['iMeans_Type_is'] )
df = pd.get_dummies(df, columns=['iRagechld'], prefix=['iRagechld_Type_is'] )
df = pd.get_dummies(df, columns=['dTravtime'], prefix=['dTravtime_Type_is'] )
df = pd.get_dummies(df, columns=['iCitizen'], prefix=['iCitizen_Type_is'] )

# Scale Data
scaled_features = df.copy()
columns = scaled_features[df.columns]
std_scale = StandardScaler().fit(columns.values)
X = std_scale.transform(columns.values)

# Perform PCA 
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
plt.plot(pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of PCA Components')
plt.ylabel('PCA Explained Variance Ratio %')
plt.title('PCA Variance Ration')
plt.show()

pca_breast = PCA(n_components=8)
principalComponents_breast = pca_breast.fit_transform(X)
principal_cencus_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8'])

print('explained_variance_ratio_', pca_breast.explained_variance_ratio_)

plt.scatter(principal_cencus_Df['principal component 1'],principal_cencus_Df['principal component 2'])
plt.title('PCA Distribution')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()

wcss=[]
for i in range(1,21):
    kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=200)
    kmeans_pca.fit(X)
    wcss.append(kmeans_pca.inertia_)

plt.plot(range(1,21),wcss, marker='o', linestyle ='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-means with PCA clustering')
plt.show()

# Selecting optimal number of clusters in KMeans
for i in range(2,20):
    labels=KMeans(n_clusters=i,init='k-means++',random_state=200).fit(X).labels_
    print('Silhouette score for k(clusters): '+str(i)+' is '+str(metrics.silhouette_score(X,labels,metric='euclidean',sample_size=1000,random_state=200)))

########################################################################## KMEANS ##########################################################################
start_kmeans = time.time()
#kmeans low performance

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(principal_cencus_Df)
principal_cencus_Df['cluster_kmeans']=y_predicted
print(principal_cencus_Df.head())
print('principal_cencus_Df.shape', principal_cencus_Df.shape)
print(km.cluster_centers_)
print(km.labels_)
print(km.inertia_)

df1 = principal_cencus_Df[principal_cencus_Df.cluster_kmeans==0]
df2 = principal_cencus_Df[principal_cencus_Df.cluster_kmeans==1]
df3 = principal_cencus_Df[principal_cencus_Df.cluster_kmeans==2]
plt.scatter(df1['principal component 1'],df1['principal component 2'],color='green', label='cluster 1')
plt.scatter(df2['principal component 1'],df2['principal component 2'],color='red', label='cluster 2')
plt.scatter(df3['principal component 1'],df3['principal component 2'],color='black', label='cluster 3')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('dAge')
plt.ylabel('dIncome1')
plt.legend()
plt.show()

# new_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(principal_cencus_Df)], axis=1)
# new_df.columns.values[-4:] = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']
# new_df['cluster']=km.labels_
# print(new_df.columns)
# print(new_df)

# df1 = new_df[new_df.cluster==0]
# df2 = new_df[new_df.cluster==1]
# df3 = new_df[new_df.cluster==2]

# plt.scatter(df1['dAge'],df1['dIncome1'],color='green', label='cluster 1')
# plt.scatter(df2['dAge'],df2['dIncome1'],color='red', label='cluster 2')
# plt.scatter(df3['dAge'],df3['dIncome1'],color='black', label='cluster 3')
# plt.xlabel('dAge')
# plt.ylabel('dIncome1')
# plt.legend()
# plt.show()

print('principal_cencus_Df.shape', principal_cencus_Df.shape)
end_kmeans = time.time()
########################################################################## Agglomerative Hierarchical Clustering #######################################################
start_hierarchy = time.time()
import scipy.cluster.hierarchy as sch
dendrogrm = sch.dendrogram(sch.linkage(principal_cencus_Df, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(principal_cencus_Df)
principal_cencus_Df['cluster_Algo'] = y_hc
principal_cencus_Df = principal_cencus_Df.values
print(y_hc)
print('X[y_hc == 0, 0]', principal_cencus_Df[y_hc == 0, 0])
print('X[y_hc == 0, 1]', principal_cencus_Df[y_hc == 0, 1])

# Visualising the clusters
plt.scatter(principal_cencus_Df[y_hc == 0, 0], principal_cencus_Df[y_hc == 0, 1], s = 50, c = 'red', label = 'Careful')
plt.scatter(principal_cencus_Df[y_hc == 1, 0], principal_cencus_Df[y_hc == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(principal_cencus_Df[y_hc == 2, 0], principal_cencus_Df[y_hc == 2, 1], s = 50, c = 'green', label = 'Target')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
end_hierarchy = time.time()
########################################################################## Gaussian Mixture Model ############################################################

start_gmm = time.time()
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
print(labels)
print('gmm.means_\n', gmm.means_)
plt.scatter(principal_cencus_Df[labels == 0, 0], principal_cencus_Df[labels == 0, 1], s = 50, c = 'red', label = 'Careful')
plt.scatter(principal_cencus_Df[labels == 1, 0], principal_cencus_Df[labels == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(principal_cencus_Df[labels== 2, 0], principal_cencus_Df[labels == 2, 1], s = 50, c = 'green', label = 'Target')
plt.title('Gaussian Mixture Model')
plt.legend()
plt.show()

# TODO: plot likelihood
end_gmm = time.time()
########################################################################## DBSCAN ##########################################################################
start_dbscan = time.time()
# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
# https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html

# Calculate the distance between 2 points
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
print('Distances', distances)
print('AVG Distances', np.average(distances))
plt.title('Calculate the distance between 2 points')
plt.plot(distances)
plt.show()

m = DBSCAN(eps=0.3, min_samples=5)
m.fit(X)

clusters = m.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)

clusters = m.labels_

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))



colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))
plt.show()

# Evaluate Model Performance — Mean Silhouette Coefficient
# https://medium.com/@tarammullin/dbscan-2788cfce9389

end_dbscan = time.time()

elapsed_time["kmeans"].append(round(end_kmeans-start_kmeans,2))
elapsed_time["gmm"].append(round(end_gmm-start_gmm,2))
elapsed_time["hierarchy"].append(round(end_hierarchy-start_hierarchy,2))
elapsed_time["dbscan"].append(round(end_dbscan-start_dbscan,2))

for x in elapsed_time:
  print('Computation Time of ' + x + ':', elapsed_time[x])
########################################################################## END ##########################################################################