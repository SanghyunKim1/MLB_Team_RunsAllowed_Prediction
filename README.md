# MLB Team Runs Allowed Prediction

## Content
1. Intro: The Nature of Baseball
2. Technologies
3. Metadata
4. Data Cleaning
5. EDA (Exploratory Data Analysis)
6. Feature Scaling
7. Multiple Linear Regression with Feature Selection
8. Simple Linear Regression
9. Model Validation
10. Conclusion

### 1. Intro
In the [previous project](https://github.com/shk204105/MLB_Team_RunsScored_Prediction), I briefly talked about how a team wins in baseball. The first part of winning in baseball is **Runs Scored (RS)** and what makes that **RS** was dealt with in the previous project using linear regression models.

However **RS** is not the only part of winning in baseball. While a team must score runs, it also has to prevent its opponents from scoring runs (at least allow runs less than it scores) to win a game. This is indicated as **Runs Allowed**. So in this project, I analyzed how a team can allow runs as less as possible.

### 2. Techonologies
- Python 3.8
  * Pandas - version 1.2.2
  * Numpy - version 1.20.1
  * matplotlib - version 3.3.4
  * seaborn - version 0.11.1
  * scikit-learn - version 0.24.1
  * statsmodels - version 0.12.2
  * scipy - version 1.6.1

### 3. Metadata
| **Metadata** | **Information** |
| :-----------: | :-----------: |
| **Origin of Data** | [Baseball Prospectus](https://www.baseballprospectus.com) |
| **Terms of Use** | [Terms and Conditions](https://www.baseballprospectus.com/terms-and-conditions/) |
| **Data Structure** | 10 datasets each consisting of 31 rows * 27 columns |

| **Data Feature** | **Data Meaning** |
| :-----------: | :-----------: |
| ***LVL*** | Level of Play: MLB (the major league) |
| ***YEAR*** | Each year refers to corresponding seasons |
| ***TEAM*** | All 30 Major League Baseball Teams |
| ***IP*** | [Innings Pitched](http://m.mlb.com/glossary/standard-stats/innings-pitched) |
| ***PA*** | [Plate Appearance](http://m.mlb.com/glossary/standard-stats/plate-appearance) |
| ***R*** | [Runs Allowed](http://m.mlb.com/glossary/standard-stats/run) |
| ***ERA*** | [Earned Run Average](http://m.mlb.com/glossary/standard-stats/earned-run-average) |
| ***FIP*** | [Fielding Independent Pitching](http://m.mlb.com/glossary/advanced-stats/fielding-independent-pitching) |
| ***cFIP*** | [Contextual Fiedling Independent Pitching](https://legacy.baseballprospectus.com/glossary/index.php?search=cFIP) |
| ***cFIP_START*** | [cFIP for Starting Pitchers](https://legacy.baseballprospectus.com/glossary/index.php?search=cFIP) |
| ***cFIP_RELIEF*** | [cFIP for Relief Pitchers](https://legacy.baseballprospectus.com/glossary/index.php?search=cFIP) |
| ***FIP_MINUS_ERA*** | [ERA Subtracted from FIP ](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=630) |
| ***SO9*** | [Strikeouts Per 9 Innings](http://m.mlb.com/glossary/advanced-stats/strikeouts-per-nine-innings) |
| ***BB9*** | [Walks Per 9 Innings](http://m.mlb.com/glossary/advanced-stats/walks-per-nine-innings) |
| ***SO/BB*** | [Strikeout-to-Walk Ratio](http://m.mlb.com/glossary/advanced-stats/strikeout-to-walk-ratio) |
| ***HR9*** | [Home Runs Per 9 Innings](http://m.mlb.com/glossary/advanced-stats/home-runs-per-nine-innings) |
| ***oppAVG*** | Batting Average Allowed by a Pitcher |
| ***oppOBP*** | On-base Percentage Allowed by a Pitcher |
| ***oppSLG*** | Slugging Percentage Allowed by a Pitcher |
| ***oppOPS*** | On-base Plus Slugging Allowed by a Pitcher |
| ***WHIP*** | [Walk and Hits Per Inning Pitched](http://m.mlb.com/glossary/standard-stats/walks-and-hits-per-inning-pitched) |
| ***DRA*** | [Deserved Run Average](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=668) |
| ***DRA-*** | [DRA-Minus](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=695) |
| ***DRA_START*** | [DRA for Starting Pitchers](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=668) |
| ***DRA_RELIEF*** | [DRA for Relief Pitchers](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=668) |
| ***PWARP*** | [Pitcher Wins Above Replacement Player](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=592) |

### 4. Data Cleaning
- Combined 10 different datasets (2010-2019 Season Pitching datasets).
- Dropped an unnecessary column made when combining datasets (Column: **'#'**).
- Renamed **'R'** data feature as **'RA'** for clarity.
- Eliminated commas in some data features and convert their data types from **integer** into **numeric** (**IP**, **PA**).
- Detected invalid **0** values in some data features (**cFIP_START**, **cFIP_RELIEF**, **SO/BB**, **oppAVG**, **oppOBP**, **oppSLG**, **oppOPS**, **DRA_START**, **DRA_RELIEF**).
- By looking at data features that contain the **0** values, I noticed that these invalid values were recorded in specific seasons because such data atrributes didn't exist in that corresponding seasson. In other words, such invalid values are considered **Missing At Random (MAR)**.
- Treated these invalid values as missing values and replaced them with projected values based on linear regression result using **IterativeImputer**. 
- Dropped categorical variables (**LVL** and **TEAM**), as they are irrelevant to this analysis.

### 5. EDA (Exploratory Data Analysis)
***5-1. RA EDA***
![](https://github.com/shk204105/MLB_Team_RunsAllowed_Prediction/blob/master/images/RA%20Histogram:Q-Q%20Plot.png)
<img src="https://github.com/shk204105/MLB_Team_RunsAllowed_Prediction/blob/master/images/Yearly%20Changes%20in%20RA.png" width="500" height="500">

- **RA** Skewness: 0.38340975864973814
- **RA** Kurtosis: -0.13976152269512854

According to the histogram and probability plot above, **RA** seems to follow a normal distribution. The skewness of 0.38 and kurtosis of -0.14 also indicate that team **RA** data is normallly distributed. Likewise, the boxplots above show that team **RA** has been normally distributed over the last 10 seasons with few outliers.


***5-2. Feature Selection: Filter Method***

<img src="https://github.com/shk204105/MLB_Team_RunsAllowed_Prediction/blob/master/images/Filtered%20Correlatoin%20Matrix.png" width="500" height="500">

| ***Correlation*** | **RS** | **PA** | **TB** | **OBP** | **ISO** | **DRC+** | **DRAA** | **BWARP** |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| **RS** | 1.0 | 0.739 | 0.922 | 0.829 | 0.812 | 0.751 | 0.806 | 0.780 |

Initially, I had 23 independent variables. To avoid multicollinearity, I filtered some of them based on (i) correlation between each **independent** variable, and (ii) correlation between those filtered features and the **dependent** variable, **RA**. As a result, I ended up **7** independent varaibles as indicated in the correlation matrix above. 


***5-3. Filtered Independent Variables EDA***

<img src="https://github.com/shk204105/MLB_Team_RunsAllowed_Prediction/blob/master/images/Histogram.png" width="800" height="800">

According to the histograms of each independent variable above, all the variables are normally distributed.

<img src="https://github.com/shk204105/MLB_Team_RunsAllowed_Prediction/blob/master/images/Scatter%20Plot.png" width="600" height="600">

Scatter plots also depict that there are reasonable linear trends between each independent variable and **RS** without notable outliers, and thus, it's safe to use the linear regression model.


### 6. Feature Scaling
Since the ranges of independent variables vary considerably, I scaled all the independent variables. As all the data attributes have normal distributions with few outliers, I used ***StandardScaler*** to scale them.

The result of feature scaling is the following:

<img src="https://github.com/shk204105/MLB_Team_RunsAllowed_Prediction/blob/master/images/KDE%20Plot.png" width="600" height="600">


### 7. Multiple Linear Regression with Feature Selection
With all the independent variables filtered above, I built a multiple linear regression model to check the degree of multicollinearity based on **VIF**.

<img width="207" alt="VIF" src="https://github.com/shk204105/MLB_Team_RunsAllowed_Prediction/blob/master/images/VIF1.png">

According to the table above, there seems to be multicollinearity in the model because independent variables are highly corrleated one another.
Therefore, I used the wrapper method (**Recursive Feature Elimination**) to find the best two independent variables.

Through **RFE**, I got **HR9** and **WHIP** as independent variables and built a multiple linear regression.
The result of the model is:

<img width="601" alt="Multiple Linear Regression" src="https://github.com/shk204105/MLB_Team_RunsAllowed_Prediction/blob/master/images/Multiple%20Linear%20Regression.png"> <img width="193" alt="VIF2" src="https://github.com/shk204105/MLB_Team_RunsAllowed_Prediction/blob/master/images/VIF2.png">


### 8. Simple Linear Regression
Apart from the multiple linear regression model, I also built a simple linear regression model. To find the sinlge best independent variable, I used the **SelectKBest** function. Based on F-statistics of each independent variable, **ERA** has beend selected as the best independent variable.

Furthermore, I also splitted data into training(70%) and test(30%) datasets for accuracy.

The result of the model is:

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Intercept*** | 44.81409069091842 |
| ***Coefficient*** | 163.46870405 |
| ***R-squared*** | 0.977594476807553 |
| ***RMSE*** | 12.37869911916763 |

As indicated in the table above, the result was TOO accurate yielding an R-squared of 0.978 and RMSE of  12.38. Such a result seems to occur because **ERA** and **RA** are almost indentical stats except the fact that **ERA (Earned Run Average)** doesn't take into account runs allowed recorded via *errors or passed plays*, while **RA** does.

In modern baseball, the quality of fielding is so outstanding compared to the past day's baseaball (imagine ball games in the 1890s or 1910s). Therefore, the odds of scoring runs with the aids of errors became so low these days. This is proven by *the correlation of 0.99* between these two stats. These two stats are almost indentical.

So although **ERA** is the best single predictor of a team's **RA**, I believe there's no point in spending time on building machine learning algorithm to just predict **RA**, if we already have **ERA**. Thus, I got the second best predictor, **WHIP**, again using **skelearn's SelectKBest**.

With **WHIP** as an independent variable the result of this model is:

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Intercept*** | -465.10977839397117 |
| ***Coefficient*** | 893.24724699 |
| ***R-squared*** | 0.7837067997149497 |
| ***RMSE*** | 38.460851534999485 |



### 9. Model Validation
<img src="https://user-images.githubusercontent.com/67542497/105632704-f1848a80-5e97-11eb-8b69-f19913f1d3be.png" width="500" height="400">

To validate both multiple and simple linear regression models, I used the K-Fold Cross Validation method, where the number of folds is 10.

***9-1. Multiple Linear Regression model validtion***

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Mean R-squared*** | 0.8905666784838221 |
| ***Mean RMSE*** | 24.9749012355517 |

***9-2. Simple Linear Regression model validtion***

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Mean R-squared*** | 0.7360878269961807 |
| ***Mean RMSE*** | 38.90144641675336 |

Accoring to the results above, the simple linear regression model (x: **WHIP** / y:**RA**) also seems to perform well. However, the accuracy is not as high as that of the multiple linear regression model (x: **HR9**, **WHIP** / y:**RA**).


### 10. Conclusion

Comparing those two models through 10-Fold Cross Validation, although **WHIP** alone is a good measure when predicting a team's **RA**, it'll result a much better result to use **HR9** and **WHIP** together for a team's **RA** prediction given the mean R-squared of **0.891** and RMSE of **24.97**.

As I mentioned in [the previous project](https://github.com/shk204105/MLB_Team_RunsScored_Prediction), a team must reach bases as many as possible to produce runs. If you're not able to reach bases, then how would you score? So if we think about it from the pitching's perspective. As a pitcher (or a team) your goal is to prevent your opponents from scoring as many as possible. How? The answer is simple. You must prevent your opponents from reaching bases by allowing as less hits, bases on balls, or hit-by-pitches as you can.

And such a job is measured by a single statistic, [WHIP](http://m.mlb.com/glossary/standard-stats/walks-and-hits-per-inning-pitched). It measures how well a pitcher has kept runners off the basepaths and is calculated by the total number of hits and walks divided by his total innings pitched. In other words, it represents how may batters a pitcher allows to reach bases per innings pitched. (e.g. a WHIP of 0.84 means that this pitcher allows 0.84 hitters to reach bases per innings he pitched)

Even though the ability to keep runners off the baspaths is still important, it seems that preventing opponent batters from reaching bases is not enough not to give up runs. There's one more thing you should do to give up as less runs as possible given my analysis. *Not allowing home runs*.

In the 2010s (especially since the 2016 season), the way a team produces runs has changed compared to the past days. Somehow, both the total number of **home runs** a team records per season, and the total number of runs created via **home runs** has inclined (see this [article](https://calltothepen.com/2019/08/29/mlb-factors-contributing-increased-home-run-rates/)).

As the way of playing the ball game changes, teams must be used to it. Therefore, from a pitching's perspective, the ability of not allowing home runs has become important these days. And such a job is measured by **HR9**, *the number of home runs allowed per 9 innings pitched*. 

To sum, if a team allows its opponents to reach bases as less as possible, and also if it allows as less home runs as it can, such a team will give up as less runs as possible. This seems to be proven through this analysis: **RA** prediction.
