
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

iCitizen
| Initial | Converted |
| --- | --- |
| 0: Born in the U.S. | 0: Born in US |
| 1: Born in Puerto Rico, Guam, and Outlying| 1: Not Born in US |
| 2: Born Abroad of American Parents | |
| 3: U.S. Citizen by Naturalization| |
| 4: Not a Citizen of the U.s| |

iClass
| Initial | Converted |
| --- | --- |
| 0: N/a Less Than 16 Yrs. Old/unemp. Who Nev| 0: Employed |
| 1: Emp. of a Private for Profit Company or| 1: not Employed|
| 2: Emp. of a Private Not for Profit, Tax Ex| |
| 3: Local Gov. Emp. City, County, Etc.| |
| 4: State Gov. Emp.| |
| 5: Fed. Gov. Emp.| |
| 6: Self Emp. in Own Not Incorp.d Business,| |
| 7: Self Emp. in Own Incorp.d Business, Prof| |
| 8: Working Without Pay in Fam. Bus. or Farm| |
| 9: Unemp., Last Worked in 1984 or Earlier| |
 
 
 
 




 # iClass
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iClass'] == 0) | (df['iClass'] == 9) , 'iClass'] = 0
df.loc[(df['iClass'] != 0) & (df['iClass'] != 9) , 'iClass'] = 1


Ability to Speak English
0:N/a Less Than 5 Yrs. Old/speaks Only Eng
1:Very Well
2:Well
3:Not Well
4:Not At All

# iEnglish      
# 0 ==> Not Speak English
# 1 ==> Speak English
df.loc[(df['iEnglish'] == 4) , 'iEnglish'] = 0
df.loc[(df['iEnglish'] >= 1) & (df['iEnglish'] <= 3) , 'iEnglish'] = 1


00:N/a Less Than 15 Yrs./male
01:No Chld.
02:1 Child
03:2 Chld.
04:3 Chld.
05:4 Chld.
06:5 Chld.
07:6 Chld.
08:7 Chld.
09:8 Chld.
10:9 Chld.
11:10 Chld.
12:11 Chld.
13:12 or More Chld.

# iFertil
# 0 ==> not 
# 1 ==> 2-4
# 2 ==> having many children 5-13
df.loc[(df['iFertil'] == 0) | (df['iFertil'] == 1) , 'iFertil'] = 0
df.loc[(df['iFertil'] >= 2) & (df['iFertil'] <= 4) , 'iFertil'] = 1
df.loc[(df['iFertil'] >= 5) & (df['iFertil'] <= 13) ,'iFertil'] = 2


HISPANIC



YEARSCH      C       X      2             Ed. Attainment
                                  00      N/a Less Than 3 Yrs. Old
                                  01      No School Completed
                                  02      Nursery School
                                  03      Kindergarten
                                  04      1st, 2nd, 3rd, or 4th Grade
                                  05      5th, 6th, 7th, or 8th Grade
                                  06      9th Grade
                                  07      10th Grade
                                  08      11th Grade
                                  09      12th Grade, No Diploma
                                  10      High School Graduate, Diploma or Ged
                                  11      Some Coll., But No Degree
                                  12      Associate Degree in Coll., Occupational
                                  13      Associate Degree in Coll., Academic Prog
                                  14      Bachelors Degree
                                  15      Masters Degree
                                  16      Professional Degree
                                  17      Doctorate Degree

# iYearsch
# 0 ==> No School Completed
# 1 ==> Median Education
# 3 ==> High Education
df.loc[(df['iYearsch'] == 0) | (df['iYearsch'] == 0) , 'iYearsch'] = 0
df.loc[(df['iYearsch'] > 2) & (df['iYearsch'] < 11) , 'iYearsch'] = 1
df.loc[(df['iYearsch'] > 10) & (df['iYearsch'] < 18) , 'iYearsch'] = 2


RSPOUSE      C       X      1             Married, Spouse Present/spouse Absent
                                  0       N/a Less Than 15 Yrs. Old
                                  1       Now Married, Spouse Present
                                  2       Now Married, Spouse Absent
                                  3       Widowed
                                  4       Divorced
                                  5       Separated
                                  6       Never Married

# iRspouse
# 0 ==> not 
# 1 ==> yes
df.loc[(df['iRspouse'] == 0) | (df['iRspouse'] == 6) , 'iRspouse'] = 0
df.loc[(df['iRspouse'] >= 1) & (df['iRspouse'] <= 5) , 'iRspouse'] = 1 


# dRearning
# 0 ==> not 
# 1 ==> earning
# 2 ==> rich
df.loc[(df['dRearning'] >= 1) & (df['dRearning'] <= 3) , 'dRearning'] = 1
df.loc[(df['dRearning'] >= 4) & (df['dRearning'] <= 5) , 'dRearning'] = 2



MEANS        C       X      2             Means of Transportation to Work
                                  00      N/a Not a Worker Not in the Labor Force,
                                  01      Car, Truck, or Van
                                  02      Bus or Trolley Bus
                                  03      Streetcar or Trolley Car
                                  04      Subway or Elevated
                                  05      Railroad
                                  06      Ferryboat
                                  07      Taxicab
                                  08      Motorcycle
                                  09      Bicycle
                                  10      Walked
                                  11      Worked At Home
                                  12      Other Method
# 0 ==> not 
# 1 ==> public transportation
# 2 ==> by o2n
# 3 ==> Worked At Home
# 4 ==> Other Method
df.loc[(df['iMeans'] => 2) | (df['iMeans'] <= 6) , 'iAvail'] = 1
df.loc[(df['iAvail'] == 1) | (df['iMeans'] => 7) | (df['iMeans'] <= 10) , 'iAvail'] = 2
df.loc[ df['iMeans'] == 11, 'iMeans'] = 3
df.loc[ df['iMeans'] == 12, 'iMeans'] = 4