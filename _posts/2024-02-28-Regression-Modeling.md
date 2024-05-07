---
title: "Chicago Bike Rentals: Regression Modeling Case Study - Kaggle Competition"
date: 2024-02-28
tags: [Python, machine learning, regression]
header:
  image: "/images/chicago-bike.jpg"
excerpt: "(Python - Machine Learning) The bike sharing industry has grown tremendously in recent years, with an estimated global value of $2.8 billion in 2023. This is due to a number of factors, such as convenience, sustainability, and physical fitness. As a result of the market's growth, your client, a major city in the United States, has tasked you with developing a machine learning model to predict the number of bike rentals on a given day, as well as to provide insights into the factors that contribute to bike rental demand. Based on a fictitious business case built by Professor Chase Kusterer from Hult International Business School"
mathjax: "true"
toc: true
toc_label : "Navigate"
---

## Chicago Bike Rentals: Regression Modeling Case Study - Kaggle Competition
By: Jorge Solis<br>
Hult International Business School<br>
<br>
<br>
Jupyter notebook and dataset for this analysis can be found here: [Portfolio](https://github.com/jorgesolisservelion/portfolio) 
<br>
<br>

***
### Introduction
The bike sharing industry has grown tremendously in recent years, with an estimated global value of $2.8 billion in 2023. This is due to a number of factors, such as convenience, sustainability, and physical fitness. As a result of the market's growth, Chicago needs to predict the number of bike rentals on a given day, as well as to get insights into the factors that contribute to bike rental demand.

### Overview
- Best performing model was a Linear Rgression with ##(number of features) features with a test score of 0.9 and a corss validation score with 11 folds at 0.9.
- Optimal features were found using regularization methods such as Lasso and ARD Regression.
- It is predicting the

***

<strong> Case - Chicago Bike Rentals. </strong> <br>
<strong>  Audience: </strong> The Cook County Planning and Development Department, responsible for the Chicago metropolitan area in the United States <br>
<strong> Goal: </strong> Predict the number of bike rentals on a given day <br>
<strong> Target consumer: </strong> Chicago citizens <br>
<strong> Product: </strong> bike rentals <br>
<strong>Channels: </strong> In person rentals <br> 

<strong>Permitted Model Types for this competition</strong><br>
<br>

| Model Type             | Method In Scikit-Learn      |
|------------------------|-----------------------------|
| OLS Linear Regression | linear_model.LinearRegression() |
| Lasso Regression       | linear_model.Lasso()         |
| Ridge Regression       | linear_model.Ridge()         |
| Elastic Net Regression| linear_model.SGDRegressor()  |
| K-Nearest Neighbors   | neighbors.KNeighborsRegressor() |
| Decision Tree Regressor | tree.DecisionTreeRegressor() |

<br><br>

***
<strong> Data and Assumptions </strong> <br>
Dataset:
<br> 2,000 customers (approx.)
- at least one purchase per month for a total of 11 of their first 12 months
- at least one purchase per quarter and at least 15 purchases through their first year
- dataset engineering techniques are statistically sound and represent the customers


Assumptions:
- all average times are in seconds
- revenue = price x quantity, where quantity is total meals ordered
- when customers attend master classes, they are looking to increase their cooking skills

***

<strong> Analysis Outline: </strong>
1. Part 1: Exploratory Data Analysis
2. Part 2: Transformations
3. Part 3: Build a machine learning model to predict the number of bike rentals
4. Part 4: Evaluating Model

*** 

### Set-up

```python
## importing libraries ##

import numpy                   as np  # mathematical essentials
import pandas                  as pd  # data science essentials
import seaborn                 as sns # enhanced graphical output
import matplotlib.pyplot       as plt # essential graphical output
import statsmodels.formula.api as smf # regression modeling
import sklearn.linear_model                               # linear models
from sklearn.model_selection import train_test_split      # train/test split
from sklearn.neighbors       import KNeighborsRegressor   # KNN for Regression
from sklearn.preprocessing   import StandardScaler        # standard scaler
from sklearn.tree            import DecisionTreeRegressor # regression trees
from sklearn.tree            import plot_tree             # plot tree
from sklearn.model_selection import RandomizedSearchCV    # hyperparameter tuning

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

## importing data ##

# reading modeling data into Python
modeling_data = './datasets/chicago_training_data.xlsx'

# calling this df_train
df_train = pd.read_excel(io         = modeling_data,
                         sheet_name = 'data',
                         header     = 0,
                         index_col  = 'ID')

# reading testing data into Python
testing_data = './datasets/test.xlsx'

# calling this df_test
df_test = pd.read_excel(io         = testing_data,
                        sheet_name = 'data',
                        header     = 0,
                        index_col  = 'ID')

# concatenating datasets together for mv analysis and feature engineering
df_train['set'] = 'Not Kaggle'
df_test ['set'] = 'Kaggle'

# concatenating both datasets together for mv and feature engineering
df_full = pd.concat(objs = [df_train, df_test],
                    axis = 0,
                    ignore_index = False)

#Renaming the columns
df_full = df_full.rename(columns={
    'Temperature(F)': 'Temperature_F',
    'Humidity(%)': 'Humidity',
    'Wind speed (mph)': 'Wind_speed',
    'Visibility(miles)': 'Visibility',
    'DewPointTemperature(F)': 'DewPointTemperature',
    'Rainfall(in)': 'Rainfall',
    'Snowfall(in)': 'Snowfall',
    'SolarRadiation(MJ/m2)': 'SolarRadiation',
})

# checking data
df_full.head(n = 5)
```

### User-defined functions
In the next section, we'll design a number of functions that will facilitate our analysis.

```python
## User-Defined Functions 

# defining a function for categorical boxplots
def categorical_boxplots(response, cat_var, data):
    """
    This function is designed to generate a boxplot for  can be used for categorical variables.
    Make sure matplotlib.pyplot and seaborn have been imported (as plt and sns).

    PARAMETERS
    ----------
    response : str, response variable
    cat_var  : str, categorical variable
    data     : DataFrame of the response and categorical variables
    """

    fig, ax = plt.subplots(figsize = (10, 8))
    
    sns.boxplot(x    = response,
                y    = cat_var,
                data = data)
    
    plt.suptitle("")
    plt.show()

# plot_feature_importances
def plot_feature_importances(model, train, export = False):
    """
    Plots the importance of features from a CART model.
    
    PARAMETERS
    ----------
    model  : CART model
    train  : explanatory variable training data
    export : whether or not to export as a .png image, default False
    """
    
    # declaring the number
    n_features = x_train.shape[1]
    
    # setting plot window
    fig, ax = plt.subplots(figsize=(12,9))
    
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')

```

***

## Part 1: Exploratory Data Analysis (EDA)

Let us first take a look at our dataset

```python
# Information about each variable
df_full.info(verbose = True)
```
Observations:
- Data types are coherent with each variable description.
- 76 missing values in Visibility, 67 in DewPointTemperature and 106 in SolarRadiation.
- Number of observations: 2184
- Total of 12 variables (including target variable) where:
    - 2 are integers
    - 7 are floats
    - 4 are objects

<br>
<strong> Step 1: </strong> Classifying our variables based on variable types:

```python

inputs_num = ['Temperature_F', 'Humidity', 'Wind_speed', 'Visibility', 'DewPointTemperature', 'Rainfall', 'Snowfall', 'SolarRadiation']
inputs_cat = ['Holiday', 'FunctioningDay']

```

<strong> Step 2: </strong> Investigating missing values

Now let's see the histogram of numerical variables
```python
# developing a histogram using HISTPLOT
sns.histplot(data  = df_train,
         x     = "RENTALS",
         kde   = True)


# title and axis labels
plt.title(label   = "Distribution of RENTALS")
plt.xlabel(xlabel = "RENTALS") # avoiding using dataset labels
plt.ylabel(ylabel = "Count")


# displaying the histogram
plt.show()
```

