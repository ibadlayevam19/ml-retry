import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#task1

df=pd.read_csv('data/titanic.csv')
df.head()
print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])
df.dtypes

#task2

# Calculate the mean of all numeric columns
df[df.select_dtypes('number').columns].mean(axis=0)
# median of all numeric columns
df[df.select_dtypes('number').columns].median(axis=0)
# Calculate the number of unique values in each column type 'object'
df[df.select_dtypes('object').columns].nunique(axis=0)

df.groupby('Sex')['Survived'].mean()
df.groupby('Pclass')['Survived'].mean()
df.groupby('Embarked')['Survived'].mean()
pd.pivot_table(df, values='Survived', index='Sex', columns='Pclass',aggfunc='mean')

#task3

df.isnull().sum()
(df.isnull().sum()/len(df))*100
#example codes for handling missing values
#df['Age'].fillna(df['Age'].median(), inplace=True)
#df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
#df.drop(columns=['Cabin'], inplace=True)

#task4

plt.hist(df['Age'], bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x=df['Age'])
plt.show()

df['Sex'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Survived(0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

df['Survived'].value_counts().plot(kind='bar')
plt.title('Survival Distribution')
plt.xlabel('Survived(0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)
plt.title('Survival Rate by Pclass and Sex')
plt.show()

pivot=pd.pivot_table(df, values='Survived', index='Sex', columns='Pclass',aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap='coolwarm')
plt.title('Survival Rate Heatmap')
plt.show()

#task5

df.drop_duplicates(inplace=True)
df['Age'].fillna(df['Age'].median(),  inplace=True)
df.isnull().sum()
df['Sex']=df['Sex'].map({'male':0, 'female':1})
df=pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df.head()

#task6

df['Survived'].mean()*100
#around 38 percent survived

df.groupby('Pclass')['Survived'].mean()
#higher Pclass had higher survival rates

df.groupby('Sex')['Survived'].mean()
#females had higher survival rates

df.groupby(pd.cut(df['Age'], bins=[0,12,20,40,60,80]))['Survived'].mean()
#children had higher survival rates