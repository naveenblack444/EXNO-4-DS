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
<img width="1156" height="534" alt="image" src="https://github.com/user-attachments/assets/c3631d08-dbde-4f20-babf-2f6b39dae4e0" />

```
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 ##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
 df[categorical_columns]
```
<img width="1131" height="518" alt="image" src="https://github.com/user-attachments/assets/cf78a570-b5c8-42d6-ac46-9e274e6833e0" />

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
<img width="857" height="110" alt="image" src="https://github.com/user-attachments/assets/a656230a-c778-4e09-a544-c5e31445af76" />

```
 y_pred = rf.predict(X_test)
```
```
 df=pd.read_csv("/content/income(1) (1).csv")
 df.info()
```
<img width="904" height="452" alt="image" src="https://github.com/user-attachments/assets/f2e7f550-0cdf-4c93-8d29-432a5c23a197" />

```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, chi2, f_classif
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```
<img width="1202" height="541" alt="image" src="https://github.com/user-attachments/assets/27d95e80-b72e-433b-abd4-be783bb0b0f6" />

```
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
<img width="1139" height="535" alt="image" src="https://github.com/user-attachments/assets/ccbbdbf2-bb07-47f6-b0de-4f346d3dff7a" />

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
```
 Selected features using chi-square test:
 Index(['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
       'hoursperweek'],
      dtype='object')
```
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
<img width="1048" height="106" alt="image" src="https://github.com/user-attachments/assets/b1c4bec1-b7dd-4e71-b17b-ffd3d1f82875" />

```
 y_pred = rf.predict(X_test)
 from sklearn.metrics import accuracy_score
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Model accuracy using selected features: {accuracy}")
```
<img width="864" height="37" alt="image" src="https://github.com/user-attachments/assets/d8ce6e23-a25e-44d9-914a-b6cde8cfc055" />

```
!pip install skfeature-chappers
```
```
Collecting skfeature-chappers
  Downloading skfeature_chappers-1.1.0-py3-none-any.whl.metadata (926 bytes)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (from skfeature-chappers) (1.6.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (from skfeature-chappers) (2.2.2)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from skfeature-chappers) (2.0.2)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas->skfeature-chappers) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas->skfeature-chappers) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas->skfeature-chappers) (2025.2)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->skfeature-chappers) (1.16.3)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->skfeature-chappers) (1.5.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->skfeature-chappers) (3.6.0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas->skfeature-chappers) (1.17.0)
Downloading skfeature_chappers-1.1.0-py3-none-any.whl (66 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.3/66.3 kB 1.6 MB/s eta 0:00:00
Installing collected packages: skfeature-chappers
Successfully installed skfeature-chappers-1.1.0
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
<img width="1145" height="535" alt="image" src="https://github.com/user-attachments/assets/b1a8a690-6705-4448-8e97-effcdca2ed0a" />

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
<img width="1284" height="80" alt="image" src="https://github.com/user-attachments/assets/d8199fa7-496b-4db2-9681-0d323e667a5c" />

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
```
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
 df[categorical_columns]
```
<img width="1326" height="530" alt="image" src="https://github.com/user-attachments/assets/c777484b-cab4-407e-877e-5bfcfe592f68" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
```
```
 logreg = LogisticRegression()
```
```
n_features_to_select =6
```
```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
 rfe.fit(X, y)
```
```
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
```
<img width="352" height="207" alt="image" src="https://github.com/user-attachments/assets/e0a5b158-babf-4348-bc51-290a7c65250e" />

     
# RESULT:
    
