---
title: "Chicago Bike Rentals: Regression Modeling Case Study"
date: 2024-02-28
tags: [Python, machine learning, regression]
header:
  image: "/images/chicago-bike.jpg"
excerpt: "(Python - Machine Learning) The bike sharing industry has grown tremendously in recent years, with an estimated global value of $2.8 billion in 2023. This is due to a number of factors, such as convenience, sustainability, and physical fitness. As a result of the market's growth, your client, a major city in the United States, has tasked you with developing a machine learning model to predict the number of bike rentals on a given day, as well as to provide insights into the factors that contribute to bike rental demand. Based on a fictitious business case built by Professor Chase Kusterer from Hult International Business School"
mathjax: "true"
toc: true
toc_label : "Navigate"
---

## Chicago Bike Rentals: Regression Modeling Case Study
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
# Importing Necessary Libraries
import pandas                  as pd  # data science essentials
import matplotlib.pyplot       as plt # essential graphical output
import seaborn                 as sns # enhanced graphical output
import statsmodels.formula.api as smf # regression modeling
from   sklearn.model_selection import train_test_split    # train test split
from   sklearn.neighbors       import KNeighborsRegressor # KNN for Regression
from   sklearn.preprocessing   import StandardScaler      # standard scaler
from   sklearn.linear_model    import LinearRegression    # linear regression (scikit-learn)
from   sklearn.model_selection import cross_val_score     # cross-validation 
import sklearn
import numpy as np

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# specifying file name
file = "Apprentice_Chef_Dataset.xlsx"

# reading the file into Python
original_df = pd.read_excel(file)
chef_org.   = original_df.copy()
```

### User-defined functions
In the next section, we'll design a number of functions that will facilitate our analysis.
