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

```
 # Chi_Square
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 #In feature selection, converting columns to categorical helps certain algorithms
 # (like decision trees or chi-square tests) correctly understand and
 # process non-numeric features. It ensures the model treats these columns as categories,
 # not as continuous numerical values.
 df[categorical_columns]
```
<img width="1267" height="524" alt="image" src="https://github.com/user-attachments/assets/32ce046c-1e0e-4a77-9d1b-84f6ed3b1c68" />

```
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 ##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
 df[categorical_columns]
```
<img width="1162" height="527" alt="image" src="https://github.com/user-attachments/assets/f9577bc3-9928-4318-9fab-7c6a82072f77" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 #X contains all columns except 'SalStat' — these are the input features used to predict something.
 #y contains only the 'SalStat' column — this is the target variable you want to predict.
```
```
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 from sklearn.ensemble import RandomForestClassifier
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
```
<img width="874" height="95" alt="image" src="https://github.com/user-attachments/assets/dde73b1a-4513-412f-9eeb-94f8c5b5cd98" />

```
 y_pred = rf.predict(X_test)
```

```
df = pd.read_csv("/content/income(1) (1).csv", engine='python', on_bad_lines='skip')
df.info()
```
<img width="1087" height="454" alt="image" src="https://github.com/user-attachments/assets/fea0770f-8c7a-4a6f-af9f-ba7b7c6b137d" />

```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, chi2, f_classif
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```


```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="1164" height="530" alt="image" src="https://github.com/user-attachments/assets/1e93f7a7-9c6e-4f7b-81a0-8bf2e5800e0a" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_chi2 = 6
 selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
 X_chi2 = selector_chi2.fit_transform(X, y)
 selected_features_chi2 = X.columns[selector_chi2.get_support()]
 print("Selected features using chi-square test:")
 print(selected_features_chi2)
```
<img width="1176" height="106" alt="image" src="https://github.com/user-attachments/assets/089ef9e7-a113-4ee9-a727-498708f8e532" />

```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, chi2, f_classif
 from sklearn.model_selection import train_test_split # Importing the missing function
 from sklearn.ensemble import RandomForestClassifier
 selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
 'hoursperweek']
 X = df[selected_features]
 y = df['SalStat']
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
```
<img width="543" height="101" alt="image" src="https://github.com/user-attachments/assets/2925d8e4-c683-4a51-89eb-d45408098dfa" />

```
 y_pred = rf.predict(X_test)
 from sklearn.metrics import accuracy_score
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Model accuracy using selected features: {accuracy}")
```
<img width="1043" height="51" alt="image" src="https://github.com/user-attachments/assets/f1f18272-6c46-466f-9374-29c55954db92" />

```
 !pip install skfeature-chappers
```

```
 import numpy as np
 import pandas as pd
 from skfeature.function.similarity_based import fisher_score
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
```
```
 categorical_columns = [
 'JobType',
 'EdType',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'gender',
 'nativecountry'
 ]
 df[categorical_columns] = df[categorical_columns].astype('category')
```
```
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 # @title
 df[categorical_columns]
```

<img width="1090" height="529" alt="image" src="https://github.com/user-attachments/assets/79fa4d19-4c2b-43d7-adc8-8622cf0d8a92" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
```
```
 k_anova = 5
 selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
 X_anova = selector_anova.fit_transform(X, y)
```

```
 selected_features_anova = X.columns[selector_anova.get_support()]
```

```
 print("\nSelected features using ANOVA:")
 print(selected_features_anova)
```
<img width="1054" height="85" alt="image" src="https://github.com/user-attachments/assets/50bb789f-6a11-44a9-a144-f141e7f3aee1" />

```
 # Wrapper Method
 import pandas as pd
 from sklearn.feature_selection import RFE
 from sklearn.linear_model import LogisticRegression
 df=pd.read_csv("/content/income(1) (1).csv")
 # List of categorical columns
 categorical_columns = [
 'JobType',
 'EdType',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'gender',
 'nativecountry'
 ]
 # Convert the categorical columns to category dtype
 df[categorical_columns] = df[categorical_columns].astype('category')

```










     
# RESULT:
       # INCLUDE YOUR RESULT HERE
