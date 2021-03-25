import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# dAncstry1,dAncstry2,iDisabl1,iDisabl2,dHour89,dHours,dIncome1,dIncome2,dIncome3,dIncome4,dIncome5,dIncome6,dIncome7,dIncome8,dIndustry,dOccup,iOthrserv,dPOB,iRelat1,iRelat2,iRiders,iRlabor,iRownchld,dRpincome,iRrelchldiSubfam1,iSubfam2,iTmpabsnt,dWeek89,iWork89,iWorklwk,iYearwrk

missing_values = ['n/a', 'na', '--', '?'] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
df=pd.read_csv(r''+my_path+'\\data\\USCensus1990.data.txt', sep=',', nrows=200000, na_values=missing_values)
print(df.shape)

# Data Preprocessing 

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['iRagechld'])
# plt.title('Count the distribution of dDepart attained Categorical Column')
# plt.show()

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['dHispanic'])
# plt.title('Count the distribution of dDepart attained Categorical Column')
# plt.show()

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['dPoverty'])
# plt.title('Count the distribution of dPoverty attained Categorical Column')
# plt.show()

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['iSchool'])
# plt.title('Count the distribution of iSchool attained Categorical Column')
# plt.show()

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['iMarital'])
# plt.title('Count the distribution of iMarital attained Categorical Column')
# plt.show()

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['iRemplpar'])
# plt.title('Count the distribution of iRemplpar attained Categorical Column')
# plt.show()

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['iAvail'])
# plt.title('Count the distribution of iAvail attained Categorical Column')
# plt.show()

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['iClass'])
# plt.title('Count the distribution of iAvail attained Categorical Column')
# plt.show()

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['dDepart'])
# plt.title('Count the distribution of dDepart attained Categorical Column')
# plt.show()

# # Visualize Education attained distribution
# sns.countplot(data=df, x=df['iFertil'])
# plt.title('Count the distribution of dDepart attained Categorical Column')
# plt.show()

# iMilitary
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iMilitary'] == 0) | (df['iMilitary'] == 4) , 'iMilitary'] = 0
df.loc[(df['iMilitary'] >= 1) & (df['iMilitary'] <= 3) , 'iMilitary'] = 1

groupedByUserId  = df.groupby(['caseid']) # group rows per user_id
df.insert(0, 'military', 0 )

for index,group in groupedByUserId:
    if group['iFeb55'].values[0] == 1 | group['iKorean'].values[0] == 1 | group['iMay75880'].values[0] == 1 | group['iRvetserv'].values[0] == 1 | group['iSept80'].values[0] == 1 | group['iVietnam'].values[0] == 1 | group['iWWII'].values[0] == 1 | group['dYrsserv'].values[0] == 1 | group['iMilitary'].values[0] == 1 :
        df.loc[index,'military'] = 1

# Drop War columns 
df.drop('iFeb55', axis=1, inplace=True) 
df.drop('iKorean', axis=1, inplace=True)
df.drop('iMay75880', axis=1, inplace=True) 
df.drop('iRvetserv', axis=1, inplace=True)
df.drop('iSept80', axis=1, inplace=True)
df.drop('iVietnam', axis=1, inplace=True)
df.drop('iWWII', axis=1, inplace=True)
df.drop('dYrsserv', axis=1, inplace=True) # year of active duty
df.drop('iMilitary', axis=1, inplace=True) # Military Srvc.

df.drop('dHispanic', axis=1, inplace=True) # since tha most cases are not hispanic
df.drop(index=df[df['dPoverty'] == 0].index,    inplace=True) # drop the rows that has N/A
df.drop('iRemplpar', axis=1, inplace=True) # since iRemplpar column has many zero values
df.drop(index=df[df['iSchool']  == 0].index,     inplace=True)   # N/a Less Than 3 Yrs. Old
#df.drop(index=df[df['iYearsch'] == 0].index,    inplace=True)    # N/a Less Than 3 Yrs. Old
df.drop(index=df[df['iEnglish'] == 0].index,    inplace=True)    # N/a Less Than 5 Yrs. Old/speaks Only Eng
df.drop(index=df[df['iImmigr'] == 0].index,    inplace=True)     # Born in the U.S. since we can take this value from the citizen column
df.drop('caseid', axis=1, inplace=True)
# YEARWRK ==> Never Worked

# Visualize military attained distribution
sns.countplot(data=df, x=df['military'])
plt.title('Count the distribution of military attained Categorical Column')
plt.show()

#TODO
#dDepart 
#Vehicle Occupancy RIDERS
#dYrsserv vs iMilitary
#iRiders
#TODO

# iLang1
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iLang1'] == 0) | (df['iLang1'] == 2) , 'iLang1'] = 0

#iMobility
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iMobility'] == 0) | (df['iMobility'] == 2) , 'iMobility'] = 0

#iMobillim
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iMobility'] == 0) | (df['iMobility'] == 2) , 'iMobility'] = 0

# iFertil
# 0 ==> not 
# 1 ==> 2-4
# 2 ==> having many children 5-13
df.loc[(df['iFertil'] == 0) | (df['iFertil'] == 1) , 'iFertil'] = 0
df.loc[(df['iFertil'] >= 2) & (df['iFertil'] <= 4) , 'iFertil'] = 1
df.loc[(df['iFertil'] >= 5) & (df['iFertil'] <= 13) ,'iFertil'] = 2

# iRspouse
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iRspouse'] == 0) | (df['iRspouse'] == 6) , 'iRspouse'] = 0
df.loc[(df['iRspouse'] >= 1) & (df['iRspouse'] <= 5) , 'iRspouse'] = 1

# iPerscare
# PERSCARE     C       X      1             Personal Care Limitation
#                                   0       N/a Less Than 15 Yrs./instit. Person, an
#                                   1       Yes, Has a Personal Care Limitation
#                                   2       No, Does Not Have a Personal Care Limita
# 0 ==> not 
# 1 ==> yes
df.loc[df['iPerscare'] == 2, 'iPerscare'] = 0

# dRearning
# 0 ==> not 
# 1 ==> earning
# 2 ==> rich
df.loc[(df['dRearning'] >= 1) & (df['dRearning'] <= 3) , 'dRearning'] = 1
df.loc[(df['dRearning'] >= 4) & (df['dRearning'] <= 5) , 'dRearning'] = 2

# dPwgt1 => Pers. Wgt
# 0 ==> slim 
# 1 ==> normal
# 2 ==> obese
df.loc[(df['dPwgt1'] == 2) | (df['dPwgt1'] == 3) , 'dPwgt1'] = 2

# iMeans
# 0 ==> not 
# 1 ==> public transportation
# 2 ==> by on vichele
# 3 ==> other
df.loc[(df['iMeans'] >= 2) | (df['iMeans'] <= 6) , 'iMeans'] = 1
df.loc[(df['iMeans'] == 1) | (df['iMeans'] >= 7) | (df['iMeans'] <= 10) , 'iMeans'] = 2
df.loc[ df['iMeans'] == 11, 'iMeans'] = 3
df.loc[ df['iMeans'] == 12, 'iMeans'] = 4

# iLooking
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iLooking'] == 0) | (df['iLooking'] == 2) , 'iLooking'] = 0
df.loc[ df['iLooking'] == 1, 'iLooking'] = 1

# iClass
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iClass'] == 0) | (df['iClass'] == 9) , 'iClass'] = 0
df.loc[(df['iClass'] != 0) & (df['iClass'] != 9) , 'iClass'] = 1

# iAvail
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iAvail'] >= 0) & (df['iAvail'] <= 3) , 'iAvail'] = 0
df.loc[ df['iAvail'] == 4, 'iAvail'] = 1

# iSchool
# 0 ==> not attend
# 1 ==> attend
df.loc[ df['iSchool'] == 1, 'iSchool'] = 0
df.loc[(df['iSchool'] >= 2) & (df['iSchool'] <= 3) , 'iSchool'] = 1

# iImmigr
# 0 ==> Came to US before 1950
# 1 ==> Came to US after 1950
df.loc[(df['iImmigr'] >= 1) & (df['iImmigr'] <= 9) , 'iImmigr'] = 0
df.loc[df['iImmigr'] == 10, 'iImmigr'] = 1

# iMarital
# 0 ==> Never Married
# 1 ==> Married 
df.loc[(df['iMarital'] >= 0) & (df['iMarital'] <= 3) , 'iMarital'] = 1
df.loc[df['iMarital'] == 4, 'iMarital'] = 0

# iYearsch
# 0 ==> No School Completed
# 1 ==> Median Education
# 3 ==> High Education
df.loc[(df['iYearsch'] == 0) | (df['iYearsch'] == 0) , 'iYearsch'] = 0
df.loc[(df['iYearsch'] > 2) & (df['iYearsch'] < 11) , 'iYearsch'] = 1
df.loc[(df['iYearsch'] > 10) & (df['iYearsch'] < 18) , 'iYearsch'] = 2

# iEnglish      
# 0 ==> Not Speak English
# 1 ==> Speak English
df.loc[(df['iEnglish'] == 4) , 'iEnglish'] = 0
df.loc[(df['iEnglish'] >= 1) & (df['iEnglish'] <= 3) , 'iEnglish'] = 1

# iRagechld
# 0 ==> No 
# 1 ==> yes
df.loc[(df['iYearsch'] == 0) | (df['iYearsch'] == 4) , 'iYearsch'] = 0
df.loc[(df['iYearsch'] >= 1) & (df['iYearsch'] <= 3) , 'iYearsch'] = 1

# dTravtime
# 0 ==> No 
# 1 ==> below 1 hour
# 2 ==> above 1 hour
df.loc[(df['iYearsch'] >= 1) & (df['iYearsch'] <= 5) , 'iYearsch'] = 1
df.loc[(df['iYearsch'] == 6), 'iYearsch'] = 2

# iCitizen
# 0 ==> Born in U.S.
# 1 ==> Born not in U.S.
df.loc[(df['iCitizen'] == 4) , 'iCitizen'] = 0
df.loc[(df['iCitizen'] >= 1) & (df['iCitizen'] <= 3) , 'iCitizen'] = 1

# Visualize Education attained distribution
sns.countplot(data=df, x=df['iRPOB'])
plt.title('Count the distribution of iRPOB attained Categorical Column')
plt.show()

# iRPOB
# 0 ==> Citizen
# 1 ==> Not a Citizen
df.loc[df['iRPOB'] != 52, 'iRPOB'] = 0
df.loc[df['iRPOB'] == 52, 'iRPOB'] = 1

# Visualize Education attained distribution
sns.countplot(data=df, x=df['iRPOB'])
plt.title('Count the distribution of iRPOB attained Categorical Column')
plt.show()

# Visualize Education attained distribution
sns.countplot(data=df, x=df['iSchool'])
plt.title('Count the distribution of iSchool attained Categorical Column')
plt.show()

# Visualize Education attained distribution
sns.countplot(data=df, x=df['iYearsch'])
plt.title('Count the distribution of Education attained Categorical Column')
plt.show()

# Visualize Marital Status distribution
sns.countplot(data=df, x=df['iMarital'])
plt.title('Count the distribution of Marital Status Categorical Column')
plt.show()

# Visualize Immigration year distribution
sns.countplot(data=df, x=df['iImmigr'])
plt.title('Count the distribution of Immigration year Categorical Column')
plt.show()

# Visualize English proficiency distribution
sns.countplot(data=df, x=df['iEnglish'])
plt.title('Count the distribution of English proficiency Categorical Column')
plt.show()

# Visualize Sex distribution
sns.countplot(data=df, x=df['iSex'])
plt.title('Count the distribution of males/females where 0:Male 1:Female')
plt.show()

# Visualize Age distribution
sns.countplot(data=df, x=df['dAge'])
plt.title('Count the distribution of dAge')
plt.show()

# Visualize Income distribution
sns.countplot(data=df, x=df['dIncome1'])
plt.title('Count the distribution of Income')
plt.show()

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

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(principal_cencus_Df)
principal_cencus_Df['cluster']=y_predicted
print(principal_cencus_Df.head())
print('principal_cencus_Df.shape', principal_cencus_Df.shape)
print(km.cluster_centers_)
print(km.inertia_)

df1 = principal_cencus_Df[principal_cencus_Df.cluster==0]
df2 = principal_cencus_Df[principal_cencus_Df.cluster==1]
df3 = principal_cencus_Df[principal_cencus_Df.cluster==2]
plt.scatter(df1['principal component 1'],df1['principal component 2'],color='green', label='cluster 1')
plt.scatter(df2['principal component 1'],df2['principal component 2'],color='red', label='cluster 2')
plt.scatter(df3['principal component 1'],df3['principal component 2'],color='black', label='cluster 3')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('dAge')
plt.ylabel('dIncome1')
plt.legend()
plt.show()

new_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(principal_cencus_Df)], axis=1)
new_df.columns.values[-4:] = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']
new_df['cluster']=km.labels_
print(new_df.columns)
print(new_df)

df1 = new_df[new_df.cluster==0]
df2 = new_df[new_df.cluster==1]
df3 = new_df[new_df.cluster==2]
df4 = new_df[new_df.cluster==3]

plt.scatter(df1['dAge'],df1['dIncome1'],color='green', label='cluster 1')
plt.scatter(df2['dAge'],df2['dIncome1'],color='red', label='cluster 2')
plt.scatter(df3['dAge'],df3['dIncome1'],color='black', label='cluster 3')
plt.scatter(df4['dAge'],df4['dIncome1'],color='yellow', label='cluster 4')
plt.xlabel('dAge')
plt.ylabel('dIncome1')
plt.legend()
plt.show()

print('principal_cencus_Df.shape', principal_cencus_Df.shape)
# Hierarchy 
import scipy.cluster.hierarchy as sch
dendrogrm = sch.dendrogram(sch.linkage(principal_cencus_Df, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(principal_cencus_Df)
principal_cencus_Df = principal_cencus_Df.values
print(y_hc)
print('X[y_hc == 0, 0]', principal_cencus_Df[y_hc == 0, 0])
print('X[y_hc == 0, 1]', principal_cencus_Df[y_hc == 0, 1])

# Visualising the clusters
plt.scatter(principal_cencus_Df[y_hc == 0, 0], principal_cencus_Df[y_hc == 0, 1], s = 50, c = 'red', label = 'Careful')
plt.scatter(principal_cencus_Df[y_hc == 1, 0], principal_cencus_Df[y_hc == 1, 1], s = 50, c = 'blue', label = 'Standard')
plt.scatter(principal_cencus_Df[y_hc == 2, 0], principal_cencus_Df[y_hc == 2, 1], s = 50, c = 'green', label = 'Target')
plt.scatter(principal_cencus_Df[y_hc == 3, 0], principal_cencus_Df[y_hc == 3, 1], s = 50, c = 'cyan', label = 'Careless')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()