import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import os

df_columns =["AAGE","AANCSTR1 ","AANCSTR2","AAUGMENT","ABIRTHPL","ACITIZEN","ACLASS","ADEPART","ADISABL1","ADISABL2","AENGLISH","AFERTIL","Age","AHISPAN","AHOUR89","AHOURS","AIMMIGR","AINCOME1","AINCOME2","AINCOME3 ","AINCOME4","AINCOME5","AINCOME6","AINCOME7","AINCOME8 ","AINDUSTR","ALABOR","ALANG1","ALANG2","ALSTWRK","AMARITAL","AMEANS ","AMIGSTAT","AMOBLLIM ","AMOBLTY","ANCSTRY1","ANCSTRY2","AOCCUP","APERCARE","APOWST","ARACE ","ARELAT1","ARIDERS ","ASCHOOL","ASERVPER ","ASEX","ATRAVTME","AVAIL","AVETS1","AWKS89","AWORK89","AYEARSCH","AYRSSERV","CITIZEN","CLASS","DEPART","DISABL1","DISABL2","ENGLISH","Feb-55","FERTIL","HISPANIC","HOUR89","HOURS","IMMIGR","INCOME1","INCOME2","INCOME3","INCOME4 ","INCOME5","INCOME6","INCOME7","INCOME8","INDUSTRY ","KOREAN","LANG1 ","LANG2 ","LOOKING","MARITAL","MAY75880 ","MEANS","MIGPUMA","MIGSTATE","MILITARY","MOBILITY","MOBILLIM","OCCUP","OTHRSERV ","PERSCARE","POB","POVERTY","POWPUMA","POWSTATE ","PWGT1","RACE","RAGECHLD","REARNING","RECTYPE ","RELAT1","RELAT2","REMPLPAR","RIDERS","RLABOR","ROWNCHLD","RPINCOME","RPOB","RRELCHLD","RSPOUSE","RVETSERV","SCHOOL","Sep-80","SERIALNO","SEX","SUBFAM1","SUBFAM2","TMPABSNT","TRAVTIME","VIETNAM","WEEK89","WORK89 ","WORKLWK ","WWII ","YEARSCH","YEARWRK","YRSSERV"]
missing_values = ["n/a", "na", "--", "?"] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
df=pd.read_csv(r''+my_path+'\\data\\USCensus1990raw.data.txt', sep='\t', nrows=200000,na_values=missing_values, names=df_columns)

# a. Data description & Visualization that aids the comprehension of the problem.
# b. Data pre-processing.
# c. Data/feature selection/evaluation.
# d. Decide how to split the data between training and data set.
# e. Use multiple classifiers and evaluate the parameters of each classifier: Try at least the following:
# Support Vector Machines (linear, and non-linear), Decision Trees, Naïve Bayes, and one based
# on ensemble learning (especially consider the Random Forests) and Neural Networks.
# f. Use clustering algorithms, evaluate parameters. Try at least the following: k-means, DBSCAN,
# agglomerative (hierarchical) clustering.
# g. In regression: Try at least linear regression, polynomial regression, and a regression algorithm of
# your choice. Explore regularization.
# h. Evaluate
# a. the performance of each classifier: at least provide F1 measure, precision, recall and
# ROC curves (if applicable)
# b. clusters based on criteria such as silhouette, and inertia.
# c. regression based on criteria such as the R score and others.
# i. Observe finding and draw conclusions.
# j. Future work: Also include things you might try/consider in the future.


# a. Sex
# b. Number of children
# c. Education attained
# d. Marital Status
# e. English proficiency
# f. Age
# g. Race
# h. Immigration year

# Age:12,INCOME1:65,RACE:95,RPOB:105,YEARSCH:122,SCHOOL:109, SEX:112, MARITAL:78, IMMIGR:64, ENGLISH:58
df =df.iloc[:, [12,65,94,109,122,112,78,64,58]]

# # Transform Categorical Data
df = pd.get_dummies(df, columns=["RACE"], prefix=["RACE_Type_is"] )
# df = pd.get_dummies(df, columns=["RPOB"], prefix=["RPOB_Type_is"] )
df = pd.get_dummies(df, columns=["YEARSCH"], prefix=["YEARSCH_Type_is"] )
df = pd.get_dummies(df, columns=["SCHOOL"], prefix=["SCHOOL_Type_is"] )
df = pd.get_dummies(df, columns=["SEX"], prefix=["SEX_Type_is"] )
df = pd.get_dummies(df, columns=["MARITAL"], prefix=["MARITAL_Type_is"] )
df = pd.get_dummies(df, columns=["IMMIGR"], prefix=["IMMIGR_Type_is"] )
df = pd.get_dummies(df, columns=["ENGLISH"], prefix=["ENGLISH_Type_is"] )
print(df)



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
plt.show()

pca_breast = PCA(n_components=33)
principalComponents_breast = pca_breast.fit_transform(X)
principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5', 'principal component 5'])

print('explained_variance_ratio_', pca_breast.explained_variance_ratio_)

plt.scatter(principal_breast_Df['principal component 1'],principal_breast_Df['principal component 2'])
plt.title('PCA Distribution')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()

# Selecting optimal number of clusters in KMeans
for i in range(2,10):
    labels=cluster.KMeans(n_clusters=i,init="k-means++",random_state=200).fit(X).labels_
    print('Silhouette score for k(clusters): '+str(i)+' is '+str(metrics.silhouette_score(X,labels,metric='euclidean',sample_size=1000,random_state=200)))


km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(principal_breast_Df)
principal_breast_Df['cluster']=y_predicted
print(principal_breast_Df.head())

print(km.cluster_centers_)
print(km.inertia_)

df1 = principal_breast_Df[principal_breast_Df.cluster==0]
df2 = principal_breast_Df[principal_breast_Df.cluster==1]
df3 = principal_breast_Df[principal_breast_Df.cluster==2]
df4 = principal_breast_Df[principal_breast_Df.cluster==3]
plt.scatter(df1['principal component 1'],df1['principal component 2'],color='green')
plt.scatter(df2['principal component 1'],df2['principal component 2'],color='red')
plt.scatter(df3['principal component 1'],df3['principal component 2'],color='black')
plt.scatter(df4['principal component 1'],df4['principal component 2'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.show()

new_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(principal_breast_Df)], axis=1)
new_df.columns.values[-4:] = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']
new_df['cluster']=km.labels_
print(new_df.columns)
print(new_df)

df1 = new_df[new_df.cluster==0]
df2 = new_df[new_df.cluster==1]
df3 = new_df[new_df.cluster==2]
df4 = new_df[new_df.cluster==3]

plt.scatter(df1['Age'],df1['INCOME1'],color='green')
plt.scatter(df2['Age'],df2['INCOME1'],color='red')
plt.scatter(df3['Age'],df3['INCOME1'],color='black')
plt.scatter(df4['Age'],df4['INCOME1'],color='yellow')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.show()

# print(df['Age'].describe())
# # count    10000.000000
# # mean        35.906600
# # std         22.425146
# # min          0.000000
# # 25%         17.000000
# # 50%         34.000000
# # 75%         52.000000
# # max         90.000000

# avgIncomeFrom10Until20=df.loc[ ( df['Age'] > 10) & (df['Age'] < 20) , 'INCOME1'].mean()
# avgIncomeFrom20Until30=df.loc[ ( df['Age'] >= 20) & (df['Age'] < 30) , 'INCOME1'].mean()
# avgIncomeFrom30Until40=df.loc[ ( df['Age'] >= 30) & (df['Age'] < 40) , 'INCOME1'].mean()
# avgIncomeFrom40Until50=df.loc[ ( df['Age'] >= 40) & (df['Age'] < 50) , 'INCOME1'].mean()
# avgIncomeFrom50Until60=df.loc[ ( df['Age'] >= 50) & (df['Age'] < 60) , 'INCOME1'].mean()
# avgIncomeFrom60Until70=df.loc[ ( df['Age'] >= 60) & (df['Age'] < 70) , 'INCOME1'].mean()
# avgIncomeFrom70Until80=df.loc[ ( df['Age'] >= 70) & (df['Age'] < 80) , 'INCOME1'].mean()

# plt.bar(x=['10-20','20-30','30-40','40-50','50-60','60-70','70-80'],
#         height=[avgIncomeFrom10Until20,avgIncomeFrom20Until30,avgIncomeFrom30Until40,avgIncomeFrom40Until50,avgIncomeFrom50Until60,avgIncomeFrom60Until70,avgIncomeFrom70Until80])
# plt.title('Average income by Age')
# plt.show()
# # print(df.shape)
# # df.drop(['caseid'],axis=1, inplace=True)

# sns.countplot(data=df, x=df['SEX'])
# plt.title("Count the distribution of males/females where 0:Male 1:Female")
# plt.show()


# # 0 Male 
# # 1 female
# maleAvgIncome=df.loc[df['SEX'] == 0, 'INCOME1'].mean()
# femaleAvgIncome=df.loc[df['SEX'] == 1, 'INCOME1'].mean()

# # Create bar char
# plt.bar(x=['MALE','FEMALE'], height=[maleAvgIncome,femaleAvgIncome])
# plt.title("Average income by SEX")
# plt.show()

# # since the list is huge
# mostFrequentValues=df['RACE'].value_counts()[:10].index.tolist()
# print(mostFrequentValues)

# whiteAvgIncome                  =df.loc[df['RACE']==1   , 'INCOME1'].mean()
# blackAvgIncome                  =df.loc[df['RACE']==2   , 'INCOME1'].mean()
# cheyenneAvgIncome               =df.loc[df['RACE']==304 , 'INCOME1'].mean()
# asianIndianAvgIncome            =df.loc[df['RACE']==10  , 'INCOME1'].mean()
# otherRaceAvgIncome              =df.loc[df['RACE']==37  , 'INCOME1'].mean()
# chineseExceptTaiwaneseAvgIncome =df.loc[df['RACE']==6   , 'INCOME1'].mean()
# koreanAvgIncome                 =df.loc[df['RACE']==11  , 'INCOME1'].mean()
# tribeNotSpecifiedAvgIncome      =df.loc[df['RACE']==327 , 'INCOME1'].mean()
# samoanAvgIncome                 =df.loc[df['RACE']==26  , 'INCOME1'].mean()
# filipinoAvgIncome               =df.loc[df['RACE']==8   , 'INCOME1'].mean()

# # Ceate Bar Char
# plt.bar(x=['White','Black','Cheyenne','Asian Indian','Other Race','Chinese','Korean','Tribe Not Specified','Samoan','Filipino'], 
#        height=[whiteAvgIncome,blackAvgIncome,cheyenneAvgIncome,asianIndianAvgIncome,otherRaceAvgIncome,chineseExceptTaiwaneseAvgIncome ,koreanAvgIncome,tribeNotSpecifiedAvgIncome,samoanAvgIncome,filipinoAvgIncome           ])
# plt.title('Avg income by race')
# plt.xticks(rotation=45)
# plt.show()


# # RPOB = Place of Birth Recode
# # 10 Born in State of Res.
# # 21 Northeast
# # 22 Midwest
# # 23 South
# # 24 West
# # 31 Puerto Rico
# # 32 American Samoa
# # 33 Guam
# # 34 Northern Marianas
# # 35 Us Virgin Islands
# # 36 Elsewhere
# # 40 Born Abroad of American Parents
# # 51 Naturalized Citizen
# # 52 Not a Citizen

# bornInStateofResAvgIncome                =df.loc[df['RPOB']==10  , 'INCOME1'].mean()
# northeastAvgIncome                       =df.loc[df['RPOB']==21  , 'INCOME1'].mean()
# midwestAvgIncome                         =df.loc[df['RPOB']==22  , 'INCOME1'].mean()
# southAvgIncome                           =df.loc[df['RPOB']==23  , 'INCOME1'].mean()
# westAvgIncome                            =df.loc[df['RPOB']==24  , 'INCOME1'].mean()
# puertoRicoAvgIncome                      =df.loc[df['RPOB']==31  , 'INCOME1'].mean()
# americanSamoaAvgIncome                   =df.loc[df['RPOB']==32  , 'INCOME1'].mean()
# guamAvgIncome                            =df.loc[df['RPOB']==33  , 'INCOME1'].mean()
# northernMarianasAvgIncome                =df.loc[df['RPOB']==34  , 'INCOME1'].mean()
# usVirginIslandsAvgIncome                 =df.loc[df['RPOB']==35  , 'INCOME1'].mean()
# elsewhereAvgIncome                       =df.loc[df['RPOB']==36  , 'INCOME1'].mean()
# bornAbroadofAmericanParentsAvgIncome     =df.loc[df['RPOB']==40  , 'INCOME1'].mean()
# naturalizedCitizenAvgIncome              =df.loc[df['RPOB']==51  , 'INCOME1'].mean()
# notaCitizenAvgIncome                     =df.loc[df['RPOB']==52  , 'INCOME1'].mean()

# plt.bar(x=['Born in State of Res.','Northeast','Midwest','South','West','Puerto Rico','American Samoa','Guam','Northern Marianas','Us Virgin Islands','Elsewhere','Born Abroad of American Parents','Naturalized Citizen','Not a Citizen'],
#         height=[bornInStateofResAvgIncome,northeastAvgIncome,midwestAvgIncome,southAvgIncome,westAvgIncome,puertoRicoAvgIncome,americanSamoaAvgIncome,guamAvgIncome,northernMarianasAvgIncome,usVirginIslandsAvgIncome,elsewhereAvgIncome,bornAbroadofAmericanParentsAvgIncome,naturalizedCitizenAvgIncome,notaCitizenAvgIncome])
# plt.title('Average by Place of Birth')
# plt.xticks(rotation=45)
# plt.show()

# # YEARSCH = Ed. Attainment
# # 00 N/a Less Than 3 Yrs. Old
# # 01 No School Completed
# # 02 Nursery School
# # 03 Kindergarten
# # 04 1st, 2nd, 3rd, or 4th Grade
# # 05 5th, 6th, 7th, or 8th Grade
# # 06 9th Grade
# # 07 10th Grade
# # 08 11th Grade
# # 09 12th Grade, No Diploma
# # 10 High School Graduate, Diploma or Ged
# # 11 Some Coll., But No Degree
# # 12 Associate Degree in Coll., Occupational
# # 13 Associate Degree in Coll., Academic Prog
# # 14 Bachelors Degree
# # 15 Masters Degree
# # 16 Professional Degree
# # 17 Doctorate Degree

# n_aLessThan3YrsOld                      =df.loc[df['YEARSCH']==0   , 'INCOME1'].mean()
# noSchoolCompleted                       =df.loc[df['YEARSCH']==1   , 'INCOME1'].mean()
# nurserySchool                           =df.loc[df['YEARSCH']==2   , 'INCOME1'].mean()
# kindergarten                            =df.loc[df['YEARSCH']==3   , 'INCOME1'].mean()
# o1st2nd3rdor4thGrade                    =df.loc[df['YEARSCH']==4   , 'INCOME1'].mean()
# o5th6th7thor8thGrade                    =df.loc[df['YEARSCH']==5   , 'INCOME1'].mean()
# othGrade                                =df.loc[df['YEARSCH']==6   , 'INCOME1'].mean()
# o10thGrade                              =df.loc[df['YEARSCH']==7   , 'INCOME1'].mean()
# o11thGrade                              =df.loc[df['YEARSCH']==8   , 'INCOME1'].mean()
# o12thGradeNoDiploma                     =df.loc[df['YEARSCH']==9   , 'INCOME1'].mean()
# highSchoolGraduateDiplomaorGed          =df.loc[df['YEARSCH']==10  , 'INCOME1'].mean()
# someCollButNoDegree                     =df.loc[df['YEARSCH']==11  , 'INCOME1'].mean()
# associateDegreeinCollOccupational       =df.loc[df['YEARSCH']==12  , 'INCOME1'].mean()
# associateDegreeinCollAcademicProg       =df.loc[df['YEARSCH']==13  , 'INCOME1'].mean()
# bachelorsDegree                         =df.loc[df['YEARSCH']==14  , 'INCOME1'].mean()
# mastersDegree                           =df.loc[df['YEARSCH']==15  , 'INCOME1'].mean()
# professionalDegree                      =df.loc[df['YEARSCH']==16  , 'INCOME1'].mean()
# doctorateDegree                         =df.loc[df['YEARSCH']==17  , 'INCOME1'].mean()

# plt.bar(x=['N/a Less Than 3 Yrs. Old','No School Completed','Nursery School','Kindergarten','1st, 2nd, 3rd, or 4th Grade','5th, 6th, 7th, or 8th Grade','9th Grade','10th Grade','11th Grade','12th Grade, No Diploma','High School Graduate, Diploma or Ged','Some Coll., But No Degree','Associate Degree in Coll., Occupational','Associate Degree in Coll., Academic Prog','Bachelors Degree','Masters Degree','Professional Degree','Doctorate Degree'],
#         height=[n_aLessThan3YrsOld,noSchoolCompleted,nurserySchool,kindergarten,o1st2nd3rdor4thGrade,o5th6th7thor8thGrade,othGrade,o10thGrade,o11thGrade,o12thGradeNoDiploma,highSchoolGraduateDiplomaorGed,someCollButNoDegree,associateDegreeinCollOccupational,associateDegreeinCollAcademicProg,bachelorsDegree,mastersDegree,professionalDegree,doctorateDegree])
# plt.title('Average by Education Attainment')
# plt.xticks(rotation=45)
# plt.show()

# # SCHOOL = School Enrollment
# # 0 N/a Less Than 3 Yrs. Old
# # 1 Not Attending School
# # 2 Yes, Pub. School, Pub. Coll.
# # 3 Yes, Private School, Private Coll.

# n_aLessThan3YrsOld              =df.loc[df['SCHOOL']==0  , 'INCOME1'].mean()
# notAttendingSchool              =df.loc[df['SCHOOL']==1  , 'INCOME1'].mean()
# yesPubSchoolPubColl             =df.loc[df['SCHOOL']==2  , 'INCOME1'].mean()
# yesPrivateSchoolPrivateColl     =df.loc[df['SCHOOL']==3  , 'INCOME1'].mean()

# plt.bar(x=['N/a Less Than 3 Yrs. Old','Not Attending School','Yes, Pub. School, Pub. Coll.','Yes, Private School, Private Coll.'],
#         height=[n_aLessThan3YrsOld,notAttendingSchool,yesPubSchoolPubColl,yesPrivateSchoolPrivateColl])
# plt.title('Average income by School Enrollment')
# plt.xticks(rotation=45)
# plt.show()

# # CITIZEN = Citizenship
# # 0 Born in the U.S.
# # 1 Born in Puerto Rico, Guam, and Outlying
# # 2 Born Abroad of American Parents
# # 3 U.S. Citizen by Naturalization
# # 4 Not a Citizen of the U.s

# bornintheUS                         =df.loc[df['CITIZEN']==0  , 'INCOME1'].mean()
# bornInPuertoRicoGuamandOutlying     =df.loc[df['CITIZEN']==1  , 'INCOME1'].mean()
# bornAbroadofAmericanParents         =df.loc[df['CITIZEN']==2  , 'INCOME1'].mean()
# uSCitizenbyNaturalization           =df.loc[df['CITIZEN']==3  , 'INCOME1'].mean()
# notaCitizenoftheUs                  =df.loc[df['CITIZEN']==4  , 'INCOME1'].mean()

# plt.bar(x=['Born in the U.S.','Born in Puerto Rico, Guam, and Outlying','Born Abroad of American Parents','U.S. Citizen by Naturalization','Not a Citizen of the U.s'],
#         height=[bornintheUS,bornInPuertoRicoGuamandOutlying,bornAbroadofAmericanParents,uSCitizenbyNaturalization,notaCitizenoftheUs])
# plt.title('Average Income by Citizebship')
# plt.xticks(rotation=45)
# plt.show()

# # CLASS = Class of Worker
# # 0 N/a Less Than 16 Yrs. Old/unemp. Who Nev
# # 1 Emp. of a Private for Profit Company or
# # 2 Emp. of a Private Not for Profit, Tax Ex
# # 3 Local Gov. Emp. City, County, Etc.
# # 4 State Gov. Emp.
# # 5 Fed. Gov. Emp.
# # 6 Self Emp. in Own Not Incorp.d Business,
# # 7 Self Emp. in Own Incorp.d Business, Prof
# # 8 Working Without Pay in Fam. Bus. or Farm
# # 9 Unemp., Last Worked in 1984 or Earlier