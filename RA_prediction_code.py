import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
from math import sqrt
from functools import reduce
import warnings
warnings.filterwarnings('ignore')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

### Purpose of Work: predicting a team's runs allowed based on various pitching statistics

# load datasets
pitching_2019 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2019.csv')
pitching_2018 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2018.csv')
pitching_2017 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2017.csv')
pitching_2016 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2016.csv')
pitching_2015 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2015.csv')
pitching_2014 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2014.csv')
pitching_2013 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2013.csv')
pitching_2012 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2012.csv')
pitching_2011 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2011.csv')
pitching_2010 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsAllowed_Prediction/data/pitching_2010.csv')


### 1. Data Cleaning ###
# merge datasets
pitching_dfs = [pitching_2019, pitching_2018, pitching_2017, pitching_2016, pitching_2015,
                pitching_2014, pitching_2013, pitching_2012, pitching_2011, pitching_2010]
pitching_df = reduce(lambda x, y: pd.merge(x, y, how='outer'), pitching_dfs)
print(pitching_df.head().to_string())

# drop unnecessary columns
pitching_df.drop(['#'], axis=1, inplace=True)

# rename specific column names
pitching_df.rename(columns={'R': 'RA'}, inplace=True)

# check data types
print(pitching_df.dtypes)

# categorical data
obj_cols = list(pitching_df.select_dtypes(include='object').columns)
print(pitching_df[obj_cols].head())

# remove commas in 'IP' and 'PA' data
cols = ['IP', 'PA']
pitching_df[cols] = pitching_df[cols].replace(',', '', regex=True)
print(pitching_df[cols].head())

# convert 'IP' and 'PA' data types from categorical data into numeric data
pitching_df[cols] = pitching_df[cols].apply(pd.to_numeric)

# drop categorical variables as we're not going to use them to predict 'RA'
pitching_df = pitching_df.select_dtypes(exclude='object')

# check new data types
print(pitching_df.dtypes)

# check missing values
print("Total Number of Missing Values in Pitching Data:")
print(pitching_df.isnull().sum())

# check duplicates
print("Total Number of Duplicates in Pitching Data: {}".format(pitching_df.duplicated().sum()))

# pitching statistics descriptive summary
print("------- Pitching Data Descriptive Summary -------")
print(pitching_df.describe().to_string())
# according to the descriptive summaries above, some data features have 0 values (invalid values)
# treat these invalid values as missing values

# check invalid '0' values
print('------- Number of 0 values in each Data Variable -------')
print(pitching_df[pitching_df == 0].count())

cols_with0 = [col for col in pitching_df.columns if any(pitching_df[col] == 0)]

# number of '0' values in each season
seasonal_df = pitching_df.groupby('YEAR')
seasonal_0cnt = seasonal_df[cols_with0].apply(lambda x: x[x == 0].count())
print(seasonal_0cnt.to_string())
# looking at the table above, the 2 '0' values in 'FIP_MINUS_ERA' data column seems reasonable
# therefore, don't treat those 2 '0' values as invalid values

# treat all the '0' values as missing values except 'FIP_MINUS_ERA' data feature
pitching_df = pitching_df.drop(['FIP_MINUS_ERA'], axis=1).replace(0, np.nan)
print(pitching_df.isnull().sum())

# Imputation
imputer = IterativeImputer(random_state=0).fit_transform(pitching_df)

pitching_df = pd.DataFrame(data=imputer, columns=pitching_df.columns)
print(pitching_df.isnull().sum())

# check imputed data decriptive summaries
print(pitching_df.describe().to_string())



### 2. EDA (Exploratory Data Analysis) ###

# dependent variable, 'RA', EDA
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.histplot(pitching_df['RA'], kde=True, ax=axes[0])
axes[0].set_title('Team RA Histogram')

stats.probplot(pitching_df['RA'], plot=axes[1])
plt.title('Team RA Q-Q Plot')

plt.show()

print('------- Team Runs Allowed Distribution -------')
print('Mean RA: {}'.format(pitching_df['RA'].mean()))
print('Median RA: {}'.format(pitching_df['RA'].median()))
print('RA Standard Deviation: {}'.format(pitching_df['RA'].std()))
print('RA Skewness: {}'.format(pitching_df['RA'].skew()))
print('RA Kurtosis: {}'.format(pitching_df['RA'].kurt()))

print('#Conclusion: Team RA distribution is approximately normal')

# yearly changes in RA
fig, ax = plt.subplots(figsize=(10, 10))

sns.boxplot(pitching_df['YEAR'], pitching_df['RA'], ax=ax)
ax.set_title('Yearly Changes in Team Runs Allowed')

plt.show()

# correlation matrix
corrMatrix= pitching_df.corr()
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrMatrix, square=True, cmap='YlGnBu', vmax=1, vmin=-1, ax=ax)
ax.set_title('Correlation Matrix')

plt.show()

print(corrMatrix.to_string())

# feature selection: filter method
# drop independent variables if its correlation between other independent variables are higher than 0.95
corrMatrix = abs(pitching_df.corr())
upperTri = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))
vars_drop = [col for col in upperTri.columns if any(upperTri[col] > 0.95)]

df = pitching_df.drop(vars_drop, axis=1)

# drop variables that have lower correlation with 'RA' than 0.65
corrMatrix = abs(df.corr())
cols = list(corrMatrix.columns)
for col in cols:
    if corrMatrix[col]['RA'] < 0.65:
        vars_drop = col
        df.drop(vars_drop, axis=1, inplace=True)

filtered_vars = list(df.columns)
print('Filtered Features: {}'.format(filtered_vars))

df = df[filtered_vars]

# new correlation matrix for selected data features
corrMatrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrMatrix, square=True, annot=True, annot_kws={'size':10}, cmap='YlGnBu',
            vmax=1, vmin=-1, center=0,
            xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns, ax=ax)
ax.set_title('Correlation Matrix')

plt.show()


# independent variables EDA
# histograms
cols = list(df.iloc[:, df.columns != 'RA'])
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for col, ax in zip(cols, axes.flatten()[:8]):
    sns.histplot(df[col], kde=True, color='blue', ax=ax)
    ax.set_title('Team {} Histogram'.format(col))

plt.show()

# Q-Q plots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for col, ax in zip(cols, axes.flatten()[:8]):
    stats.probplot(df[col], plot=ax)
    ax.set_title('{} Q-Q Plot'.format(col))

plt.show()

# scatter plots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for col, ax in zip(cols, axes.flatten()[:8]):
    sns.regplot(x=col, y='RA', data=df,
                scatter_kws={'color':'navy'}, line_kws={'color':'red'}, ax=ax)
    ax.set_title('Correlation between Team {} and RA'.format(col))

plt.show()



### 3. Feature Scaling ###
print('------- Pitching Statistics Descriptive Summary -------')
print(pitching_df.describe().to_string())
# since data ranges vary considerably scale them using StandardScaler

# StandardScaler
scaled_df = df.drop(['RA'], axis=1)
cols = list(scaled_df.columns)

std_scaler = StandardScaler()
scaled_data = std_scaler.fit_transform(scaled_df)
scaled_df = pd.DataFrame(scaled_data, columns=cols)

# KDE plot after Scaling
scaled_cols = list(scaled_df.columns)
fig, ax = plt.subplots(figsize=(8, 8))

for col in scaled_cols:
    sns.kdeplot(scaled_df[col], label=col, ax=ax)
    ax.set_title('After StandardScaler')
    ax.set_xlabel('Data Scale')
    plt.legend(loc=1)

plt.show()



### 4. Multiple Linear Regression with feature selection
# check multicollinearity
df = pd.concat([pitching_df['RA'], scaled_df], axis=1)

x = df.iloc[:, df.columns != 'RA']
x = sm.add_constant(x)
y = df['RA']

lm = sm.OLS(y, x)
result = lm.fit()
print(result.summary())

# Variance Inflation Factor (VIF)
vif = pd.DataFrame()
vif['Feature'] = lm.exog_names
vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# Recursive Feature Elimination
cols = list(x.columns)
lm = LinearRegression()

rfe = RFE(lm, 2)
x_rfe = rfe.fit_transform(x, y)
lm.fit(x_rfe, y)
temp = pd.Series(rfe.support_, index=cols)
selected_vars = list(temp[temp == True].index)

print('Selected Features: {}'.format(selected_vars))

# check VIF
x = df[selected_vars]
x = sm.add_constant(x)
y = df['RA']

lm = sm.OLS(y, x)
result = lm.fit()
print(result.summary())

vif = pd.DataFrame()
vif['Feature'] = lm.exog_names
vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
print(vif[vif['Feature'] != 'const'])

# split data into training (70%) and test(30%) data
# multiple linear regression (x: 'WHIP', 'HR9' / y: 'RA')
x = df[selected_vars]
y = df['RA']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

lm = linear_model.LinearRegression().fit(x_train, y_train)
y_predict = lm.predict(x_test)

print('------- Multiple Linear Regression -------')
print('------- Intercept -------')
print(lm.intercept_)

print('------- Coefficient -------')
print(lm.coef_)

print('------- R-squared -------')
print(metrics.r2_score(y_test, y_predict))

print('------- RMSE -------')
mse = metrics.mean_squared_error(y_test, y_predict)
print(sqrt(mse))



### 5. Simple Linear Regression ###
# univariate feature selection
x = pitching_df.iloc[:, pitching_df.columns != 'RA']
y = pitching_df['RA']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

selector = SelectKBest(score_func=f_regression, k=1)
selected_xTrain = selector.fit_transform(x_train, y_train)
selected_xTest = selector.transform(x_test)

all_cols = x.columns
selected_mask = selector.get_support()
selected_var = all_cols[selected_mask].values

print('Selected Feature: {}'.format(selected_var))

# simple linear regression (x: 'ERA' / y: 'RA')
x = pitching_df[selected_var]
y = pitching_df['RA']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

lm = linear_model.LinearRegression().fit(x_train, y_train)
y_predict = lm.predict(x_test)

print('------- Simple Linear Regression -------')
print('------- Intercept -------')
print(lm.intercept_)

print('------- Coefficient -------')
print(lm.coef_)

print('------- R-squared -------')
print(metrics.r2_score(y_test, y_predict))

print('------- RMSE -------')
mse = metrics.mean_squared_error(y_test, y_predict)
print(sqrt(mse))
# since the accuracy of the model is too high, find the second best predictor

# feature selection for the second best predictor
x = pitching_df.drop(['ERA', 'RA'], axis=1)
y = pitching_df['RA']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

selector = SelectKBest(score_func=f_regression, k=1)
selected_xTrain = selector.fit_transform(x_train, y_train)
selected_xTest = selector.transform(x_test)

all_cols = x.columns
selected_mask = selector.get_support()
selected_var = list(all_cols[selected_mask].values)

print('Selected Second Best Feature: {}'.format(selected_var))

# simple linear regression (x: 'WHIP' / y:'RA')
x = pitching_df[selected_var]
y = pitching_df['RA']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

lm = linear_model.LinearRegression().fit(x_train, y_train)
y_predict = lm.predict(x_test)

print('------- Simple Linear Regression -------')
print('------- Intercept -------')
print(lm.intercept_)

print('------- Coefficient -------')
print(lm.coef_)

print('------- R-squared -------')
print(metrics.r2_score(y_test, y_predict))

print('------- RMSE -------')
mse = metrics.mean_squared_error(y_test, y_predict)
print(sqrt(mse))



### 6. Model Validation
# 10-Fold Cross Validation for the multiple linear regression model
model = LinearRegression()

x = df[selected_vars]
y = df['RA']

cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
cv_rmse = np.sqrt(-1 * cv_mse)

print('------- Multiple Linear Regression Validation -------')
print('Mean R-squared: {}'.format(cv_r2.mean()))
print('Mean RMSE: {}'.format(cv_rmse.mean()))

# 10-Fold Cross Validation for the simple linear regression model
model = LinearRegression()

x = pitching_df[selected_var]
y = pitching_df['RA']

cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
cv_rmse = np.sqrt(-1 * cv_mse)

print('------- Simple Linear Regression Validation -------')
print('Mean R-squared: {}'.format(cv_r2.mean()))
print('Mean RMSE: {}'.format(cv_rmse.mean()))
