import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import warnings
import os

# Import Data
my_path = os.path.abspath(os.path.dirname(__file__))
missing_values = ["n/a", "na", "--", "?"] # pandas only detect NaN, NA,  n/a and values and empty shell
df=pd.read_csv(r''+my_path+'\\data\\USCensus1990.data.txt', sep=',', nrows=200000, na_values=missing_values)
df.drop('caseid', axis=1, inplace=True)
for column in df:
    plt.boxplot(df[column])
    plt.title(column)
    plt.show()


categorical_features = df.select_dtypes(include=['object', 'category']).columns # get all the categorical features
print(categorical_features.shape)

for feature in categorical_features:
    sns.countplot(data=df, x=df[feature])
    plt.title('Count the distribution of '+ feature + ' Feature')
    plt.show()

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_features = df.select_dtypes(include=numerics).columns # get all the numeric features
print(numeric_features.shape)

for feature in numeric_features:
    plt.boxplot(df[feature])
    plt.title(feature)
    plt.show()