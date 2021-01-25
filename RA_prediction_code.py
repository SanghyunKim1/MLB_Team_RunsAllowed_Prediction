import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
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
pitching_df = reduce(lambda left,right: pd.merge(left, right, how='outer'), pitching_dfs)
print(pitching_df.head().to_string())

# drop unnecessary columns
pitching_df.drop(['#'], axis=1, inplace=True)

# rename specific column names
pitching_df.rename(columns={'R': 'RA'}, inplace=True)

# check missing values
print("Total Number of Missing Values in Pitching Data:")
print(pitching_df.isnull().sum())

# check duplicates
print("Total Number of Duplicates in Pitching Data: {}".format(pitching_df.duplicated().sum()))


### 2. EDA (Exploratory Data Analysis) ###
print("------- Pitching Data Descriptive Summary -------")
print(pitching_df.describe().to_string())

## RA data distribution
print('------- Team Runs Allowed Distribution -------')
sns.displot(pitching_df['RA'], kde=True)
plt.title('Team RA Distribution')
plt.tight_layout()
plt.show()

stats.probplot(pitching_df['RA'], plot=plt)
plt.title('Team RA Probability Plot')
plt.tight_layout()
plt.show()

print('Mean RA: {}'.format(pitching_df['RA'].mean()))
print('Median RA: {}'.format(pitching_df['RA'].median()))
print('RA Standard Deviation: {}'.format(pitching_df['RA'].std()))
print('RA Skewness: {}'.format(pitching_df['RA'].skew()))
print('RA Kurtosis: {}'.format(pitching_df['RA'].kurt()))

print('#Conclusion: Team RA distribution is approximately normal')

# yearly changes in RA
yearly_ra = pd.concat([pitching_df['YEAR'], pitching_df['RA']], axis=1)

fig, ax = plt.subplots(figsize=(10, 10))

sns.boxplot(x='YEAR', y='RA', data=yearly_ra, ax=ax)
ax.set(title='Yearly Changes in Team Runs Allowed')

plt.show()

# pitching data correlation heatmap
corrP = pitching_df.corr()
sns.heatmap(corrP, square=True, cmap='Blues_r')
plt.title('Pitching Statistics Correlation Heatmap')
plt.tight_layout()

plt.show()

# Top10 RA correlation heatmap
top10_corrP_cols = corrP.nlargest(10, 'RA')['RA'].index
top10_corrP = np.corrcoef(pitching_df[top10_corrP_cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(top10_corrP, cmap='Blues_r', cbar=True, annot=True, square=True, annot_kws={'size':10},
            xticklabels=top10_corrP_cols.values, yticklabels=top10_corrP_cols.values)
plt.title("Top10 Pitching Statistics Correlated with RA")
plt.tight_layout()

plt.show()

# pitching statistics visualization
high_corrP = pitching_df.reindex(columns=top10_corrP_cols)
x_axesP = list(high_corrP.columns[1:10])

## histograms
fig, axes = plt.subplots(3, 3, figsize=(20, 20))

for col, ax in zip(x_axesP, axes.flatten()[:9]):
    sns.histplot(x=pitching_df[col], color='blue', kde=True, ax=ax)
    ax.set_title('Team {} Histogram'.format(col))
    ax.set_xlabel('Team {}'.format(col))
    ax.set_ylabel('Count')

plt.show()

## scatter plots
fig = plt.figure(figsize=(20, 20))
cScatter = 1

for col in x_axesP:
    plt.subplot(3, 3, cScatter)
    plt.title("Correlation between Team {} and RA".format(col))
    plt.xlabel(col)
    plt.ylabel("RA")
    plt.scatter(x=col, y="RA", data=high_corrP, c=high_corrP[col])
    cScatter = cScatter + 1

plt.show()

# multicollinearity
data = pitching_df[top10_corrP_cols].drop(['RA', 'DRA_START'], axis=1)
cols = list(data.columns)

x = pitching_df[cols]
x = sm.add_constant(x)
y = pitching_df['RA']

lm = sm.OLS(y, x)
result = lm.fit()
print(result.summary())

# VIF
vif = pd.DataFrame()
vif['Feature'] = lm.exog_names
vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
print(vif.loc[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# feature selection
print('------- Pitching Statistics Correlation -------')
print(pitching_df.corr().to_string())
feature_selec = pitching_df[cols].drop(['DRA-', 'FIP', 'ERA', 'DRA', 'BB9', 'cFIP'], axis=1)
cols = list(feature_selec.columns)

x = pitching_df[cols]
x = sm.add_constant(x)
y = pitching_df['RA']

lm = sm.OLS(y, x)
result = lm.fit()
print(result.summary())

vif = pd.DataFrame()
vif['Feature'] = lm.exog_names
vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
print(vif.loc[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# feature scaling
# check the scale of 'WHIP' and 'HR9'
fig, ax = plt.subplots(figsize=(10, 10))
sns.kdeplot(pitching_df['WHIP'], ax=ax, label='WHIP')
sns.kdeplot(pitching_df['HR9'], ax=ax, label='HR9')
ax.set_title('Before RobustScaler')
ax.set_xlabel('Data Scale')
ax.legend(loc='upper right')

plt.show()

# scale data
RB_scaler = preprocessing.RobustScaler()
scaled_df = RB_scaler.fit_transform(pitching_df[['WHIP', 'HR9']])
scaled_df = pd.DataFrame(scaled_df, columns=['WHIP', 'HR9'])

# check scaled 'WHIP' and 'HR9' KDE plot
fig, ax = plt.subplots(figsize=(10, 10))
cols = list(scaled_df.columns)

for col in cols:
    sns.kdeplot(scaled_df[col], ax=ax, label=col)
    ax.set_title('After RobustScaler')
    ax.set_xlabel('Data Scale')
    ax.legend(loc=1)

plt.show()

# multiple linear regression
data = pd.concat([scaled_df[['WHIP', 'HR9']], pitching_df['RA']], axis=1)
x = data[['WHIP', 'HR9']]
x = sm.add_constant(x)
y = data['RA']

lm = sm.OLS(y, x)
result = lm.fit()
print(result.summary())

x = data[['WHIP', 'HR9']]
y = data['RA']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

lm = linear_model.LinearRegression().fit(x_train, y_train)
y_predicted = lm.predict(x_test)

print('------- Intercept -------')
print(lm.intercept_)

print('------- Coefficient -------')
print(lm.coef_)

print('------- RMSE -------')
mse = metrics.mean_squared_error(y_test, y_predicted)
print(sqrt(mse))

print('------- R-squared -------')
print(metrics.r2_score(y_test, y_predicted))

# K-Fold Cross-Validation
model = LinearRegression()
cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
cv_rmse = np.sqrt(-1 * cv_mse)
print('------- 10-Fold Cross-Validation -------')
print('Mean R-squared: {}'.format(cv_r2.mean()))
print('Mean RMSE: {}'.format(cv_rmse.mean()))


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

