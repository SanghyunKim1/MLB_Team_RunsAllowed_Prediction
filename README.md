# MLB Team Runs Allowed Prediction

## Content
1. Intro: The Nature of Baseball
2. Metadata
3. Data Cleaning
4. EDA (Exploratory Data Analysis)
5. Feature Scaling
6. Multiple Linear Regression with Feature Selection
7. Simple Linear Regression
8. Model Validation
9. Conclusion

### 1. Intro
In the [previous project](https://github.com/shk204105/MLB_Team_RunsScored_Prediction), I briefly talked about how a team wins in baseball. The first part of winning in baseball is **Runs Scored (RS)** and what makes that **RS** was dealt with in the previous project using linear regression models.

However **RS** is not the only part of winning in baseball. While a team must score runs, it also has to prevent your opponents from scoring runs (at least allow runs less than it scores) to win a game. This is indicated as **Runs Allowed**. So in this project, I analyzed how a team can allow runs as less as possible.

### 2. Metadata
| **Metadata** | **Information** |
| :-----------: | :-----------: |
| **Origin of Data** | [Baseball Prospectus](https://www.baseballprospectus.com) |
| **Terms of Use** | [Terms and Conditions](https://www.baseballprospectus.com/terms-and-conditions/) |
| **Data Structure** | 10 datasets each consisting of 31 rows * 27 columns |

| **Data Feature** | **Data Meaning** |
| :-----------: | :-----------: |
| ***LVL*** | Level of Play: MLB (the major league) |
| ***YEAR*** | Each year refers to corresponding seasons |
| ***TM*** | All 30 Major League Baseball Teams |
| ***IP*** | Number of games played in the corresponding season |
| ***PA*** | [Plate Appearance](http://m.mlb.com/glossary/standard-stats/plate-appearance) |
| ***R*** | [At-bat](http://m.mlb.com/glossary/standard-stats/at-bat) |
| ***ERA*** | [Runs Scored](http://m.mlb.com/glossary/standard-stats/run) |
| ***FIP*** | [Total Bases](http://m.mlb.com/glossary/standard-stats/total-bases) |
| ***cFIP*** | [Hit](http://m.mlb.com/glossary/standard-stats/hit) |
| ***cFIP_START*** | [Batting Average](http://m.mlb.com/glossary/standard-stats/batting-average) |
| ***cFIP_RELIEF*** | [On-base Percentage](http://m.mlb.com/glossary/standard-stats/on-base-percentage) |
| ***FIP_MINUS_ERA*** | [Slugging Percentage](http://m.mlb.com/glossary/standard-stats/slugging-percentage) |
| ***SO9*** | [On-base Plus Slugging](http://m.mlb.com/glossary/standard-stats/on-base-plus-slugging) |
| ***BB9*** | [Isolated Power](http://m.mlb.com/glossary/advanced-stats/isolated-power) |
| ***SO/BB*** | [Home run](http://m.mlb.com/glossary/standard-stats/home-run) |
| ***HR9*** | [Home Run Rate](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=344) |
| ***oppAVG*** | [Walk](http://m.mlb.com/glossary/standard-stats/walk) |
| ***oppOBP*** | [Walk Rate](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=283) |
| ***oppSLG*** | [Strikeout](http://m.mlb.com/glossary/standard-stats/strikeout) |
| ***oppOPS*** | [Strikeout Rate](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=349) |
| ***WHIP*** | [Strikeout-to-Walk ratio (equivalently K/BB)](http://m.mlb.com/glossary/advanced-stats/strikeout-to-walk-ratio) |
| ***DRA*** | [Stolen Base](http://m.mlb.com/glossary/standard-stats/stolen-base) |
| ***DRA-*** | [Caught Stealing](http://m.mlb.com/glossary/standard-stats/caught-stealing) |
| ***DRA_START*** | [Stolen-base Percentage](http://m.mlb.com/glossary/standard-stats/stolen-base-percentage) |
| ***DRA_RELIEF*** | [Deserved Runs Created for a batter](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=696) |
| ***PWARP*** | [Deserved Runs Above Average](https://legacy.baseballprospectus.com/glossary/index.php?search=DRC_RAA) |

### 3. Data Cleaning
- Combined 10 different datasets (2010-2019 Season Batting datasets).
- Dropped an unnecessary column made when combining datasets (Column: **'#'**).
- Renamed **'R'** data feature as **'RS'** for clarity.
- Eliminated commas in some data features and convert their data types from **integer** into **numeric** (**'PA'**, **'AB'**, **'H'**, **'SO'**).
- Confirmed that there are no missing values and duplicates.
- Dropped categorical variables (**LG** and **TEAM**), as they are not related to our analysis.

### 4. EDA (Exploratory Data Analysis)
***4-1. RS EDA***
![RS Histogram:Probability Plot](https://user-images.githubusercontent.com/67542497/105629056-1cfd7a00-5e84-11eb-9166-ebbd49161ed1.png)
<img src="https://user-images.githubusercontent.com/67542497/105629055-1c64e380-5e84-11eb-94bb-60ff4948660d.png" width="500" height="500">

- **RS** Skewness: 0.35264844525853095
- **RS** Kurtosis: 0.061150254394042314

According to the histogram and probability plot above, **RS** seems to follow a normal distribution. The skewness of 0.35 and kurtosis of 0.06 also indicate that team **RS** data is normallly distributed. Likewise, the boxplots above show that team **RS** has been normally distributed over the last 10 seasons with few outliers.


***4-2. Feature Selection: Filter Method***

<img src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/Filtered%20Correlation%20Matrix.png" width="500" height="500">

| ***Correlation*** | **RS** | **PA** | **TB** | **OBP** | **ISO** | **DRC+** | **DRAA** | **BWARP** |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| **RS** | 1.0 | 0.739 | 0.922 | 0.829 | 0.812 | 0.751 | 0.806 | 0.780 |

Initially, I had 24 independent variables. To avoid multicollinearity, I filtered some of them based on (i) correlation between each **independent** variable, and (ii) correlation between those filtered features and the **dependent** variable, **RS**. As a result, I ended up **7** independent varaibles as indicated in the correlation matrix above. 


***4-3. Filtered Independent Variables EDA***

<img src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/Histogram.png" width="800" height="800">

According to the histograms of each independent variable above, all the variables are normally distributed.

<img src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/Scatter%20plot.png" width="600" height="600">

Scatter plots also depict that there are reasonable linear trends between each independent variable and **RS** without notable outliers, and thus, it's safe to use the linear regression model.


### 5. Feature Scaling
Since the ranges of independent variables vary considerably, I scaled all the independent variables. As all the data attributes have normal distributions with few outliers, I used ***StandardScaler*** to scale them.

The result of feature scaling is the following:

<img src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/KDE%20plot.png" width="600" height="600">


### 6. Multiple Linear Regression with Feature Selection
With all the independent variables filtered above, I built a multiple linear regression model to check the degree of multicollinearity based on **VIF**.

<img width="207" alt="VIF" src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/VIF1.png">

According to the table above, there seems to be multicollinearity in the model because independent variables are highly corrleated one another.
Therefore, I used the wrapper method (**Recursive Feature Elimination**) to find the best two independent variables.

Through **RFE**, I got **TB** and **OBP** as independent variables and built a multiple linear regression.
The result of the model is:

<img width="601" alt="Multiple Linear Regression" src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/Multiple%20Linear%20Regression.png"> <img width="193" alt="VIF2" src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/VIF2.png">


### 7. Simple Linear Regression
Apart from the multiple linear regression model, I also built a simple linear regression model. To find the sinlge best independent variable, I used the **SelectKBest** function. Based on F-statistics of each independent variable, **OPS** has beend selected as the best independent variable.

Furthermore, I also splitted data into training(70%) and test(30%) datasets for accuracy.

The result of the model is:

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Intercept*** | -752.3837394309697 |
| ***Coefficient*** | 2009.36777208 |
| ***R-squared*** | 0.9079557000049954 |
| ***RMSE*** | 23.207166436311425 |


### 8. Model Validation
<img src="https://user-images.githubusercontent.com/67542497/105632704-f1848a80-5e97-11eb-8b69-f19913f1d3be.png" width="500" height="400">

To validate both multiple and simple linear regression models, I used the K-Fold Cross Validation method, where the number of folds is 10.

***8-1. Multiple Linear Regression model validtion***

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Mean R-squared*** | 0.8584381525775473 |
| ***Mean RMSE*** | 24.92574073069298 |

***8-2. Simple Linear Regression model validtion***

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Mean R-squared*** | 0.8610824571143239 |
| ***Mean RMSE*** | 24.37742789357559 |

Accoring to the results above, the simple linear regression model (x:**OPS** / y:**RS**) showed a slightly higher R-squared than the multiple linear regression model (x:**TB**, **OBP** / y:**RS**).
However, the differences in the R-squared between those two models are marginal, and as both models don't overfit data, it's safe to use either model to predict team **RS**.


### 9. Conclusion

Comparing those two models through 10-Fold Cross Validation, although the simple linear regression seems more accurate, the differences between these two models seem marginal.

One possible reason for such a result is because these two predictors (**OPS** vs **TB + OBP**) measure similar things in baseball. For those who are not familiar with baseball, let me briefly talk about what these three stats measure in baeball.


Frist, [**TB**](http://m.mlb.com/glossary/standard-stats/total-bases) measures total number of bases gained through hits. It assigns **1 total base** to a single, **2 total bases** to a double, **3 total bases** to a triple and **4 total bases** to a home run. Therefore, a team's TB shows how many **singles** as well as **extra hits (doubles + triples + home runs)**, which possibly advance runners on base, have been gained throughout the season by that team.

Second, [**OBP**](http://m.mlb.com/glossary/standard-stats/on-base-percentage) (On-Base Percentage) measures how *often* a batter reaches bases (e.g an **OBP** of 0.400 means that this batter has reached bases four times in 10 plate appearances). It includes *Hits (singles + extra hits)*, *Base-on-Balls* and *Hit-by-Pitches*. While **TB** measures total number of bases gained, **OBP** measures the efficiency of a batter in terms of the ability to reach bases.

Finally, [**OPS**](http://m.mlb.com/glossary/standard-stats/on-base-plus-sluggin) is the sum of **OBP** and [**SLG**](http://m.mlb.com/glossary/standard-stats/slugging-percentage). **SLG** here refers to *Slugging Percentage*. This **SLG** shows the total number of bases (**TB**) a hitter records *per at-bat*. As **SLG** doesn't include *Base-on-Balls* and *Hit-by-Pitches* (these two are measured by **OBP**), if we combine **OBP** and **SLG** together, we get a single statistic that measures similar things as **TB + OBP** does.

The nature of baseball again. As I mentioned at the beginning of this project, a team should outscore its opponents to win a game in baseball. To do so, that team has to score and it's indicated as **Runs Scored (RS)**, the dependent variable. Then how does a team score runs?

Simple. To score runs in baseball, a team's batters must reach bases (i.e. become runners on bases) and other batters must advance these runners on bases to drive runs. This is how a team scores in baseball.

And this is what either **OPS** or **TB + OBP** measure, (a) the ability to reach bases, and (b) the ability to advance runners on bases to drive runs.

Given this fact, there's no wonder that those two different models yield the similar level of accuracy. Bothe independet variables measure similar things. Therefore, we get the similar result. Although the simple linear regression model where the independent variable is **OPS** yields a marginally more accurate result, I believe we'd get similar results no matter which one we use to predict team **RS**.
