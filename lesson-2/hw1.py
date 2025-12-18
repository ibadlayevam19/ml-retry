import os
import tarfile
import urllib.request
import pandas as pd
from sklearn.impute import SimpleImputer

DOWNLOAD_ROOT = "https://github.com/ageron/data/raw/main/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

fetch_housing_data()
df = pd.read_csv(os.path.join(HOUSING_PATH, "housing/housing.csv"))


#task1

df.head(10)
df.info()
df.describe()
df[df.select_dtypes('object').columns].value_counts()
df.isna().sum()
df.dtypes

#task2

df1=df.select_dtypes('number').copy()


imp=SimpleImputer(strategy='median')
imp.fit(df1)

df2=pd.DataFrame(imp.transform(df1), columns=imp.feature_names_in_)
df2.isna().sum()

def missing_report(dataFrame):
    data=[]
    for col in dataFrame.columns:
        missing_count=dataFrame[col].isna().sum()
        if missing_count>0:
            percent_missing=(missing_count/dataFrame.shape[0])*100
            data.append([col, missing_count, percent_missing])

    return pd.DataFrame(data, columns=["Column Name", "Missing Count", "Percent Missing"])
missing_report(df)

#task3

for_encode=df.copy()
one_hot_encoded=pd.get_dummies(for_encode, columns=['ocean_proximity'], dtype=int)
one_hot_encoded.head()

#task4

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled=scaler.fit_transform(df2)
scaled_df=pd.DataFrame(scaled, columns=df2.columns)

df2['median_income'].hist()
scaled_df['median_income'].hist()
df2['housing_median_age'].hist()
scaled_df['housing_median_age'].hist()
df2['population'].hist()
scaled_df['population'].hist()
df2['median_house_value'].hist()
scaled_df['median_house_value'].hist()

from sklearn.preprocessing import MinMaxScaler
min_max_scaler=MinMaxScaler()
minmax_scaled=min_max_scaler.fit_transform(df2)
minmax_scaled_df=pd.DataFrame(minmax_scaled, columns=df2.columns)

minmax_scaled_df['median_income'].hist()
minmax_scaled_df['housing_median_age'].hist()
minmax_scaled_df['population'].hist()
minmax_scaled_df['median_house_value'].hist()

