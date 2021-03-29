
# Exploratory Data Analysis

The USCensus19910raw data set, it is a multivariate dataset which consist of 124 feature, the 54 is categorical.
https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29

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

iCitizen
Citizenship
| Initial | Converted |
| --- | --- |
| 0: Born in the U.S. | 0: Born in US |
| 1: Born in Puerto Rico, Guam, and Outlying| 1: Not Born in US |
| 2: Born Abroad of American Parents | |
| 3: U.S. Citizen by Naturalization| |
| 4: Not a Citizen of the U.s| |

iClass
Class of Worker
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
 
iEnglish
Ability to Speak English
| Initial | Converted |
| --- | --- |
| 0:N/a Less Than 5 Yrs. Old/speaks Only Eng| 0: Not Speak English |
| 1:Very Well| 1: Speak English |
| 2:Well| |
| 3:Not Well| |
| 4:Not At All| |

iFertil
No. of Chld. Ever Born
| Initial | Converted |
| --- | --- |
| 00:N/a Less Than 15 Yrs./male | 0: No Chld. |
| 01:No Chld. | 1: 2-4 Child |
| 02:1 Child | 2: 5-13 Having many Child |
| 03:2 Chld. | |
| 04:3 Chld. | |
| 05:4 Chld. | |
| 06:5 Chld. | |
| 07:6 Chld. | |
| 08:7 Chld. | |
| 09:8 Chld. | |
| 10:9 Chld. | |
| 11:10 Chld. | |
| 12:11 Chld. | |
| 13:12 or More Chld. | |

iYearsch
Ed. Attainment
| Initial | Converted |
| --- | --- |
| 00: N/a Less Than 3 Yrs. Old | 0: No School Completed |
| 01: No School Completed | 1:Median Education |
| 02: Nursery School | 2:High Education |
| 03: Kindergarten | |
| 04: 1st, 2nd, 3rd, or 4th Grade | |
| 05: 5th, 6th, 7th, or 8th Grade | |
| 06: 9th Grade | |
| 07: 10th Grade | |
| 08: 11th Grade | |
| 09: 12th Grade, No Diploma | |
| 10: High School Graduate, Diploma or Ged | |
| 11: Some Coll., But No Degree | |
| 12: Associate Degree in Coll., Occupational | |
| 13: Associate Degree in Coll., Academic Prog | |
| 14: Bachelors Degree | |
| 15: Masters Degree | |
| 16: Professional Degree | |
| 17: Doctorate Degree | |

iRspouse
Married, Spouse Present/spouse Absent
| Initial | Converted |
| --- | --- |
| 0: N/a Less Than 15 Yrs. Old | :0 No |
| 1: Now Married, Spouse Present | :1 Yes |
| 2: Now Married, Spouse Absent | |
| 3: Widowed | |
| 4: Divorced | |
| 5: Separated | |
| 6: Never Married | |

MEANS
Means of Transportation to Work 
| Initial | Converted |
| --- | --- |
| 00: N/a Not a Worker Not in the Labor Force | 0: Not|
| 01: Car, Truck, or Van | 1: Public Transportation|
| 02: Bus or Trolley Bus | 2: By own|
| 03: Streetcar or Trolley Car | 3: Worked At Home|
| 04: Subway or Elevated | 4: Other Method|
| 05: Railroad | |
| 06: Ferryboat | |
| 07: Taxicab | |
| 08: Motorcycle | |
| 09: Bicycle | |
| 10: Walked | |
| 11: Worked At Home | |
| 12: Other Method | |

MILITARY
Military Srvc.
| Initial | Converted |
| --- | --- |
| 0:N/a Less Than 16 Yrs. Old| 0: No service|
| 1:Yes, Now on Active Duty| 1: Service|
| 2:Yes, on Active Duty in Past, But Not Now| |
| 3:Yes, Srvc. in Reserves or Nat. Guard Onl| |
| 4:No Srvc.| |

YRSSERV
Yrs. of Active Duty Military Srvc.
|0 = | |
|1 < 5 | |
|2 else | |

LANG1
Language Other Than English At Home
| Initial | Converted |
| --- | --- |
| 0:N/a Less Than 5 Yrs. Old | 0: No |
| 1:Yes, Speaks Another Language | 1: Yes|
| 2:No, Speaks Only English | |

MOBILITY
Mobility Stat. Lived Here on April 1, 19
| Initial | Converted |
| --- | --- |
| 0:N/a Less Than 5 Yrs. Old | 0: No |
| 1:Yes Same House Nonmovers |  1: Yes |
| 2:No, Different House Movers | |


FERTIL
No. of Chld. Ever Born
| Initial | Converted |
| --- | --- |
| 00:N/a Less Than 15 Yrs./male | 0: No |
| 01:No Chld. | 1: 2-4|
| 02:1 Child | 2: having many children 5-13|
| 03:2 Chld. | |
| 04:3 Chld. | |
| 05:4 Chld. | |
| 06:5 Chld. | |
| 07:6 Chld. | |
| 08:7 Chld. | |
| 09:8 Chld. | |
| 10:9 Chld. | |
| 11:10 Chld. | |
| 12:11 Chld. | |
| 13:12 or More Chld. | |

RSPOUSE
Married, Spouse Present/spouse Absent
| Initial | Converted |
| --- | --- |
|0:N/a Less Than 15 Yrs. Old | 0: No|
|1:Now Married, Spouse Present | 1: Yes |
|2:Now Married, Spouse Absent | |
|3:Widowed | |
|4:Divorced | |
|5:Separated | |
|6:Never Married | |

iPerscare
Personal Care Limitation
| Initial | Converted |
| --- | --- |
|0:N/a Less Than 15 Yrs./instit. Person, an | 0: No |
|1:Yes, Has a Personal Care Limitation | 1: Yes |
|2:No, Does Not Have a Personal Care Limita | |

REARNING
Total Pers. Earnings
| Initial | Converted |
| --- | --- |
| 0 < 0 | 0: No |
| 1 < 15000 | 1: Medium Earning |
| 2 < 20000 | 2: Rich |
| 3 < 60000 | |
| 5 else | |

dPwgt1
Pers. Wgt
| Initial | Converted |
| --- | --- |
| 0 < 50 | 0: Slim  |
| 1 < 125 | 1: Normal |
| 2 < 200 | 2: Obese |
| 3 else | |

LOOKING
Looking for Work
| Initial | Converted |
| --- | --- |
| 0:N/a Less Than 16 Yrs. Old/at Work/did No |  0: No   |
| 1:Yes | 1: Yes |
| 2:No |  |

AVAIL
Available for Work
| Initial | Converted |
| --- | --- |
| 0:N/a Less Than 16 Yrs./at Work/not Lookin | 0: No |
| 1:No, Already Has a Job | 1: Yes |
| 2:No, Temply. Ill | |
| 3:No, Other Reasons in School, Etc. | |
| 4:Yes, Could Have Taken a Job | |

SCHOOL
School Enrollment
| Initial | Converted |
| --- | --- |
| 0: N/a Less Than 3 Yrs. Old | 0: Not attend |
| 1: Not Attending School | 1: Attend |
| 2: Yes, Pub. School, Pub. Coll. | |
| 3: Yes, Private School, Private Coll. | |

IMMIGR
Yr. of Entry
| Initial | Converted |
| --- | --- |
|00:Born in the U.S. |  0: Came to US before 1950 |
|01:1987 to 1990 | 1: Came to US after 1950 |
|02:1985 to 1986 | |
|03:1982 to 1984 | |
|04:1980 or 1981 | |
|05:1975 to 1979 | |
|06:1970 to 1974 | |
|07:1965 to 1969 | |
|08:1960 to 1964 | |
|09:1950 to 1959 | |
|10:Before 1950 | |

MARITAL
Marital Stat.
| Initial | Converted |
| --- | --- |
| 0:Now Married, Except Separated | 0: Never Married |
| 1:Widowed |  1: Married  |
| 2:Divorced | |
| 3:Separated | |
| 4:Never Married or Under 15 Yrs. Old | |

RAGECHLD
Presence and Age of Own Chld.
| Initial | Converted |
| --- | --- |
| 0:N/a Male | |
| 1:With Own Chld. Under 6 Yrs. Only | 0: No  |
| 2:With Own Chld. 6 to 17 Yrs. Only | 1: Yes |
| 3:With Own Chld. Under 6 Yrs. and 6 to 17 | |
| 4:No Own Chld. .incl. Females Under 16 Yrs | |

dTravtime
Temp. Absence From Work
| Initial | Converted |
| --- | --- |
| 0 = 0 | 0: No  |
| 1 < 10 | 1: Below 1 hour |
| 2 < 15 | Above 1 hour |
| 3 < 20 | |
| 4 < 30 | |
| 5 < 60 | |
| else 6 | |

RPOB
Place of Birth Recode
| Initial | Converted |
| --- | --- |
| 10: Born in State of Res. | 0: Citizen |
| 21: Northeast | 1: Not a Citizen |
| 22: Midwest | |
| 23: South | |
| 24: West | |
| 31: Puerto Rico | |
| 32: American Samoa | |
| 33: Guam | |
| 34: Northern Marianas | |
| 35: Us Virgin Islands | |
| 36: Elsewhere | |
| 40: Born Abroad of American Parents | |
| 51: Naturalized Citizen | |
| 52: Not a Citizen | |

dDepart
Time of Departure for Work Hour and Minu
| Initial | Converted |
| --- | --- |
|0 = 0 | |
|1 < 600 | |
|2 < 700 | |
|3 < 800 | |
|4 < 1000 | |
|5 else | |

RIDERS
Vehicle Occupancy
| Initial | Converted |
| --- | --- |
| 0:N/a Not a Worker or Worker Whose Means o | |
| 1:Drove Alone | |
| 2:2 People | |
| 3:3 People | |
| 4:4 People | |
| 5:5 People | |
| 6:6 People | |
| 7:7 to 9 People | |
| 8:10 or More People | |

### New Columns
We create a new column called War. This column has 2 values 0 (no), 1(yes).
This column indicates if the person takes place at least in one war. 
In order to fill the War column we use the following columns
iFeb55,iKorean,iMay75880,iRvetserv,iSept80,iVietnam,iWWII

### Drop Colums

We drop the columns (iFeb55,iKorean,iMay75880,iRvetserv,iSept80,iVietnam,iWWII) that has information about in what war a person has involved since we add this information in war column.


## Principal Components Analysis
By reducing the number of features, we’re improving the performance of our algorithm. 
we need to decide how many features we’d like to keep based on the cumulative variance plot
TODO: add plot as image

## K-Means Clustering
TODO: add elbot plot as image

## Hierachical Clustering

## DBSCAN
How to compute min_samples?
I didn't find an automatic way to compute the min_sample.
A rule of thumb in order to compute the min_sample for high dimensonal dataset is to multimply the columns by 2.
In order to calculate the distance from each point to its closest  neighbor we are using the NearestNeighbors


### Computation Time
| Model | Time (in seconds) |
| --- | --- |
| kmeans | |
| gmm | |
| hierarchy | |
| dbscan | |
