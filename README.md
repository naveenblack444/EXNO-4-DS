# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("/content/bmi.csv")
 df.head()
```
<img width="882" height="258" alt="image" src="https://github.com/user-attachments/assets/f5304681-f861-472e-b0db-aa546c8286f3" />

```
 df_null_sum=df.isnull().sum()
 df_null_sum
```
<img width="728" height="263" alt="image" src="https://github.com/user-attachments/assets/8c2735f1-d191-4e57-8deb-2af606e30f99" />

```
df.dropna()
```

<img width="856" height="527" alt="image" src="https://github.com/user-attachments/assets/84c78ce9-adfb-4beb-974a-e15cab495468" />

```
 max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
 max_vals
```

<img width="599" height="185" alt="image" src="https://github.com/user-attachments/assets/4d42797e-7d49-4d09-9925-b61f186030c8" />

```
 from sklearn.preprocessing import StandardScaler
 df1=pd.read_csv("/content/bmi.csv")
 df1.head()
```
<img width="880" height="262" alt="image" src="https://github.com/user-attachments/assets/f45263cb-08d0-4517-a28a-96ee24527280" />

```
sc = StandardScaler()
df1[['Height', 'Weight']] = sc.fit_transform(df1[['Height', 'Weight']])
df1.head(10)
```
<img width="1024" height="458" alt="image" src="https://github.com/user-attachments/assets/033c9860-cc38-412c-adc1-104cf46e2d0b" />

```
 from sklearn.preprocessing import MinMaxScaler
 scaler=MinMaxScaler()
 df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
 df.head(10)
```

<img width="1140" height="443" alt="image" src="https://github.com/user-attachments/assets/240cf08f-4b7b-4c3f-9d5c-ecb7855a2210" />

```
 from sklearn.preprocessing import MaxAbsScaler
 scaler = MaxAbsScaler()
 df3=pd.read_csv("/content/bmi.csv")
 df3.head()
 df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
 df
```

<img width="1008" height="530" alt="image" src="https://github.com/user-attachments/assets/2de1e134-e438-4e46-9fa1-30b70ff814f5" />

```
 from sklearn.preprocessing import RobustScaler
 scaler = RobustScaler()
 df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
 df3.head()
```
<img width="1068" height="268" alt="image" src="https://github.com/user-attachments/assets/f40af643-363b-479c-aacb-cdf5f6dcfe1e" />

```
df = pd.read_csv("/content/income(1) (1).csv", on_bad_lines='skip', engine='python')
df.info()
```

<img width="809" height="443" alt="image" src="https://github.com/user-attachments/assets/4051d0cb-56db-4950-a9d4-749c0c06b25b" />

```
 df_null_sum=df.isnull().sum()
 df_null_sum
```

<img width="1038" height="609" alt="image" src="https://github.com/user-attachments/assets/68367144-0d5b-4385-9a36-85217ed61aec" />


     
# RESULT:
       # INCLUDE YOUR RESULT HERE
