import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
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

# looking at data ranges, the 2 '0' values in 'FIP_MINUS_ERA' data feature seems valid
# therefore, don't treat those 2 '0' values as invalid values

# treat all the '0' values as missing values except 'FIP_MINUS_ERA' data
pitching_df = pitching_df.drop(['FIP_MINUS_ERA'], axis=1).replace(0, np.nan)
print(pitching_df.isnull().sum())

# Imputation
imputer = IterativeImputer(random_state=0).fit_transform(pitching_df)

pitching_df = pd.DataFrame(data=imputer, columns=pitching_df.columns)
print(pitching_df.isnull().sum())

# check imputed data decriptive summaries
print(pitching_df.describe().to_string())



### 2. EDA (Exploratory Data Analysis) ###

# dependent variable, 'RA' EDA
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.histplot(pitching_df['RA'], kde=True, ax=axes[0])
axes[0].set_title('Team RA Histogram')

axes[1] = stats.probplot(pitching_df['RA'], plot=plt)
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
corr = pitching_df.corr()
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr, square=True, cmap='YlGnBu', vmax=1, vmin=-1, ax=ax)
ax.set_title('Correlation Matrix')

plt.show()

print(corr.to_string())

# drop variables that have lower correlations with 'RA' than 0.65
corr = abs(pitching_df.corr())
corr_df = corr['RA'].to_frame(name='Correlation with RA').T
corr_cols = corr_df.columns

corr_df.drop(columns=corr_cols[(corr_df < 0.65).any()], inplace=True)
print(corr_df.to_string())

selected_cols = list(corr_df.columns)
pitching_df = pitching_df[selected_cols]
print(pitching_df.head().to_string())

# new correlation matrix for selected data features
corr = pitching_df.corr()
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr, square=True, annot=True, annot_kws={'size':10}, cmap='YlGnBu',
            vmax=1, vmin=-1, center=0,
            xticklabels=corr.columns, yticklabels=corr.columns, ax=ax)
ax.set_title('Correlation Matrix')

plt.show()


# independent variables EDA
# histograms
cols = pitching_df.drop(['RA'], axis=1)
fig, axes = plt.subplots(4, 3, figsize=(20, 20))

for col, ax in zip(cols, axes.flatten()[:11]):
    sns.histplot(pitching_df[col], kde=True, color='blue', ax=ax)
    ax.set_title('Team {} Histogram'.format(col))

plt.show()

# Q-Q plots
fig, axes = plt.subplots(4, 3, figsize=(21, 21))

for col, ax in zip(cols, axes.flatten()[:11]):
    stats.probplot(pitching_df[col], plot=ax)
    ax.set_title('{} Q-Q Plot'.format(col))

plt.show()

# scatter plots
fig, axes = plt.subplots(4, 3, figsize=(20, 20))

for col, ax in zip(cols, axes.flatten()[:11]):
    sns.regplot(x=col, y='RA', data=pitching_df,
                scatter_kws={'color':'navy'}, line_kws={'color':'red'}, ax=ax)
    ax.set_title('Correlation between Team {} and RA'.format(col))

plt.show()



### 3. Feature Scaling ###

print('------- Pitching Statistics Descriptive Summary -------')
print(pitching_df.describe().to_string())
# since data ranges vary considerably scale them using StandardScaler

# StandardScaler
df = pitching_df.drop(['RA'], axis=1)
cols = list(df.columns)

std_scaler = StandardScaler()
scaled_data = std_scaler.fit_transform(df)
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
df = pd.concat([pitching_df['RA'], scaled_df], axis=1)
x = df.drop(['RA'], axis=1)
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


# # exclude 'DRA-'
# x = df.drop(['RA', 'DRA-'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# # exclude 'cFIP_START' and 'DRA_START'
# x = df.drop(['RA', 'DRA-', 'cFIP_START', 'DRA_START'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# # exclude 'PWARP'
# x = df.drop(['RA', 'DRA-', 'cFIP_START', 'DRA_START', 'PWARP'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# # exclude 'FIP'
# x = df.drop(['RA', 'DRA-', 'cFIP_START', 'DRA_START', 'PWARP', 'FIP'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# # exclude 'ERA'
# x = df.drop(['RA', 'DRA-', 'cFIP_START', 'DRA_START', 'PWARP', 'FIP', 'ERA'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# # exclude 'DRA'
# x = df.drop(['RA', 'DRA-', 'cFIP_START', 'DRA_START', 'PWARP', 'FIP', 'ERA', 'DRA'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# # exclude 'cFIP'
# x = df.drop(['RA', 'DRA-', 'cFIP_START', 'DRA_START', 'PWARP', 'FIP', 'ERA', 'DRA', 'cFIP'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# # include 'DRA' again
# x = df.drop(['RA', 'DRA-', 'cFIP_START', 'DRA_START', 'PWARP', 'FIP', 'ERA', 'cFIP'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# # exclude 'DRA' and include 'ERA' again
# x = df.drop(['RA', 'DRA-', 'cFIP_START', 'DRA_START', 'PWARP', 'FIP', 'DRA', 'cFIP'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# # exclude 'ERA' and 'PA'
# x = df.drop(['RA', 'DRA-', 'cFIP_START', 'DRA_START', 'PWARP', 'FIP', 'DRA', 'cFIP', 'ERA', 'PA'], axis=1)
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
#
# x = df[['WHIP', 'HR9', 'PA']]
# x = sm.add_constant(x)
# y = df['RA']
#
# lm = sm.OLS(y, x)
# result = lm.fit()
# print(result.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))





# x = np.array(pitching_df['ERA']).reshape(-1, 1)
# y = pitching_df['RA']
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#
# lm = linear_model.LinearRegression().fit(x_train, y_train)
# y_predicted = lm.predict(x_test)
#
# print('------- Intercept -------')
# print(lm.intercept_)
#
# print('------- Coefficient -------')
# print(lm.coef_)
#
# print('------- RMSE -------')
# mse = metrics.mean_squared_error(y_test, y_predicted)
# print(sqrt(mse))
#
# print('------- R-squared -------')
# print(metrics.r2_score(y_test, y_predicted))
#
# model = LinearRegression()
# cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
# cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
# cv_rmse = np.sqrt(-1 * cv_mse)
# print('Mean R-squared: {}'.format(cv_r2.mean()))
# print('Mean RMSE: {}'.format(cv_rmse.mean()))

