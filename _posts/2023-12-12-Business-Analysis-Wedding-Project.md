---
title: "Business Analysis Project: Are Wedding Providers with Sustainable Practices More Cost Effective?"
date: 2023-12-12
tags: [business analysis, data science, sustainability, python, sql]
header:
  image: "/images/Business-Analysis-Wedding/wedding_sustainability.jpg"
excerpt: "(Business Analysis, Sustainability, SQL, Python) Businesses are currently in constant evolution, where the concept of sustainability has taken on an important role within organizations, positioning itself as a competitive advantage. More and more customers are aware of the environmental and social impacts of their purchases, actively seeking sustainable options when making purchases. This trend has strongly influenced the wedding industry, where providers have been adapting sustainability practices in their businesses, capitalizing on a growing market segment."
mathjax: "true"
toc: true
toc_label : "Navigate"
---

## Are Wedding Providers with Sustainable Practices More Cost Effective?
By: Jorge Solis<br>
Hult International Business School<br>
<br>
<br>
Jupyter notebook and dataset for this analysis can be found here: [Portfolio](https://github.com/jorgesolisservelion/portfolio) 
<br>
<br>

***
## Introduction to the Business Question

Businesses are currently in constant evolution, where the concept of sustainability has taken on an important role within organizations, positioning itself as a competitive advantage. More and more customers are aware of the environmental and social impacts of their purchases, actively seeking sustainable options when making purchases. This trend has strongly influenced the wedding industry, where providers have been adapting sustainability practices in their businesses, capitalizing on a growing market segment.

The business question is: **Are wedding providers with sustainable practices more cost effective?** Since cost data for providers is not available, this question focuses on whether wedding providers with sustainable practices are price-efficient for the end consumer.

## Definition of Key Terms

The key terms are as follows:

**Sustainable Practices**:
Definition: In the wedding industry, sustainable practices refer to the adoption of methods and processes that minimize any environmental impact, including the use of locally sourced and organic materials, efficient waste management, recycling, etc.
- Source: [UCLA Sustainability](https://www.sustain.ucla.edu/what-is-sustainability/)
- Source: [National Geographic Society](https://www.nationalgeographic.org/encyclopedia/sustainability/)

**Cost Effective**:
Definition: Within the wedding provider context, due to the lack of cost information, we refer to the ability to offer services and products at a competitive price for the customer. It is important to note that it does not solely focus on being the cheapest option but aims to strike a balance in offering better value in terms of quality, sustainability, and price.
- Source: [Kumar, A., & Mani, M.](https://www.frontiersin.org/articles/10.3389/fenrg.2019.00011/full)

## Methodology

The project is divided into two parts:
1. **Data Extraction using SQL**: This involves querying the database to retrieve the required data.
2. **Data Analysis using Python**: This involves analyzing the data, creating visualizations, and answering the business question.

***

<strong> Case - Addressing the Business Question. </strong> <br>
<strong>  Audience: </strong> Business Client <br>
<strong> Goal: </strong> Answer your business question: Are wedding vendors with sustainable practices more cost effective? <br>
<strong> Target consumer: </strong> Vendors <br>
<strong> Product: </strong> Wedding product <br>
<strong>Channels: </strong> In person/Web/Social Networks <br> 
<br><br>

***

## Data Extraction using SQL

SQL code block:
```sql
USE Wedding_database;

# We are browsing in the database
SELECT p.product_id, p.product_name, v.vendor_id, p.price_unit, p.unit_vol, p.price_ce, 
v.vendor_depart, v.vendor_name, v.vendor_location, v.vendor_sustainable, 
CASE 
    WHEN v.vendor_rating = 0 THEN ''
    ELSE v.vendor_rating
END AS rating_vendor,
fs.flower_season, fs.flower_style, vs.artificial_flowers_in_portfolio,
att.color, att.tie, att.number_of_buttons, att.lapel_style,
cat.category_name,
sus.equip_ec, sus.avg_usage_hours, sus.total_ec, sus.number_equipment
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
LEFT JOIN Flower_Season_Style as fs
ON p.product_id = fs.product_id
LEFT JOIN Flowers_Vendor_Sustainability as vs
ON v.vendor_id = vs.vendor_id
LEFT JOIN attire as att
ON p.product_id = att.product_id
LEFT JOIN categories as cat
ON p.product_id = cat.product_id
LEFT JOIN Sustainability as sus
ON v.vendor_id = sus.vendor_id;

# Obtaining the number of records by department
SELECT vendor_depart, COUNT(*) AS obs, ROUND((COUNT(*)*100/(SELECT COUNT(*) FROM Products)),2)AS perc
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY vendor_depart
ORDER BY obs desc;

# Analyzing the price unit
SELECT p.unit_vol, count(*) as obs
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY p.unit_vol
ORDER BY obs desc;

# Categorizing the units of price measurement
SELECT CASE
    WHEN p.unit_vol IN ('per person','1 per bride side','1 per bride trial','1 per groom side','1 per kids','1 per bride','1 per person') THEN 'Individual Services'
    WHEN p.unit_vol IN ('per service','1 per vendor','1','unit') THEN 'Per Unit Services'
    WHEN p.unit_vol IN ('per 100 invitations','6 hours','1 per table','1 per piece','1 per chair','1 per traditional','1 per backdrop','1 per airbrush') THEN 'Event Supplies'
    ELSE 'Others'
    END AS unit_v, count(*)
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY unit_v
ORDER BY count(*) desc;

#Now lets see location
SELECT v.vendor_location, count(*) as obs
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY v.vendor_location
ORDER BY obs desc; #We can see that we have several observations with the same name but with different capitalizations

#Lets normalize the location data
SELECT CASE
        WHEN v.vendor_location IN ('san francisco', 'san francisco ', ' san francisco ') THEN 'San Francisco'
        WHEN v.vendor_location IN ('san jose', 'san jose ') THEN 'San Jose'
        WHEN v.vendor_location IN ('oakland', 'oakland ') THEN 'Oakland'
        WHEN v.vendor_location IN ('santa clara', 'santa clara ') THEN 'Santa Clara'
        WHEN v.vendor_location IN ('berkeley', 'berkeley ') THEN 'Berkeley'
        WHEN v.vendor_location IN ('hayward', 'hayward ') THEN 'Hayward'
        WHEN v.vendor_location IN ('los gatos', 'los gatos ') THEN 'Los Gatos'
        WHEN v.vendor_location IN ('livermore ', 'livermore') THEN 'Livermore'
        WHEN v.vendor_location IN ('walnut creek ', 'walnut creek') THEN 'Walnut Creek'
        WHEN v.vendor_location IN ('sausolito', 'sausalito ','sausalito') THEN 'Sausalito'
        WHEN v.vendor_location IN ('san anselmo ', 'san alselmo') THEN 'San Alselmo'
        WHEN v.vendor_location IN ('redwood city', 'redwood') THEN 'Redwood'
        WHEN v.vendor_location IN ('fremont', 'freemont ') THEN 'Freemont'
        WHEN v.vendor_location IN ('concord ', 'concord',' concord ') THEN 'Concord'
        ELSE v.vendor_location
    END AS standardized_location,
    count(*) as obs
FROM Products as p -- We got a standard data of location
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY standardized_location
ORDER BY standardized_location desc;

#We can group the locations in areas

SELECT
    CASE
        WHEN standardized_location IN ('San Francisco', 'Oakland', 'Berkeley', 'Hayward', 'Livermore', 
                                       'Los Gatos', 'Walnut Creek', 'Concord', 'castro valley', 
                                       'san rafael', 'novato ', 'napa', 'brisbane', 
                                       'san luis obispo', 'San Alselmo', 'sacramento', 'daly city', 
                                       'morro bay', 'monterey', 'milbrae ', 'martinez', 
                                       'greenbrae', 'el cerrito ', 'alameda', 'palo alto', 
                                       'burlingame', 'sunnyvale', 'hillsborough', 'lafayette', 
                                       'san leandro', 'san carlos', 'gilroy', 'san mateo ', 
                                       'south san francisco', 'scotts valley', 'paso robles', 'petaluma', 
                                       'watsonville', 'vacaville ', 'tiburon ', 'sunol ', 'studio', 
                                       'stanford ', 'san ramon ', 'san joaquin valley', 'san diego', -- Grouping locations
                                       'saint martin', 'richmond ', 'pleasanton', 'pleasant hill', 
                                       'pittsburg', 'pescadero ', 'oakley ', 'oakley', 'nicasio ', 
                                       'menlo park ', 'mammoth lakes ', 'hollister ', 'hercules', 
                                       'felton ', 'felton', 'dublin ', 'dixon ', 'cupertino ', 
                                       'cupertino', 'corte madera', 'cloverdale ', 'clayton ', 
                                       'carmel', 'campbell ', 'campbell', 'calistoga ', 'brentwood ', 
                                       'belmont', 'antioch', 'acampo ') THEN 'San Francisco Bay Area'
        WHEN standardized_location IN ('San Jose', 'Santa Clara') THEN 'South Bay'
        WHEN standardized_location IN ('Sausalito', 'Tiburon') THEN 'North Bay'
        WHEN standardized_location IN ('Fremont', 'Livermore', 'Pleasanton ', 'mountain view ', 'half moon bay ') THEN 'East Bay'
        WHEN standardized_location IN ('palo alto', 'Redwood', 'Menlo Park') THEN 'Peninsula'
        WHEN standardized_location = 'online' THEN 'Online' -- We group locations
        ELSE 'Other'
    END AS region_group,
    count(*) as obs
FROM (SELECT CASE -- We need to create a subquery 
        WHEN v.vendor_location IN ('san francisco', 'san francisco ', ' san francisco ') THEN 'San Francisco'
        WHEN v.vendor_location IN ('san jose', 'san jose ') THEN 'San Jose'
        WHEN v.vendor_location IN ('oakland', 'oakland ') THEN 'Oakland'
        WHEN v.vendor_location IN ('santa clara', 'santa clara ') THEN 'Santa Clara'
        WHEN v.vendor_location IN ('berkeley', 'berkeley ') THEN 'Berkeley'
        WHEN v.vendor_location IN ('hayward', 'hayward ') THEN 'Hayward' 
        WHEN v.vendor_location IN ('los gatos', 'los gatos ') THEN 'Los Gatos'
        WHEN v.vendor_location IN ('livermore ', 'livermore') THEN 'Livermore'
        WHEN v.vendor_location IN ('walnut creek ', 'walnut creek') THEN 'Walnut Creek'
        WHEN v.vendor_location IN ('sausolito', 'sausalito ','sausalito') THEN 'Sausalito'
        WHEN v.vendor_location IN ('san anselmo ', 'san alselmo') THEN 'San Alselmo'
        WHEN v.vendor_location IN ('redwood city', 'redwood') THEN 'Redwood'
        WHEN v.vendor_location IN ('fremont', 'freemont ') THEN 'Freemont'
        WHEN v.vendor_location IN ('concord ', 'concord',' concord ') THEN 'Concord'
        ELSE v.vendor_location
    END AS standardized_location
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY standardized_location
ORDER BY standardized_location desc) as SUBQUERY -- this is the subquery
GROUP BY region_group
ORDER BY region_group desc;

#Join and create a table
SELECT p.product_id, p.product_name, v.vendor_id, p.price_unit, p.unit_vol, p.price_ce, 
v.vendor_depart, v.vendor_name, v.vendor_location, 
CASE
        WHEN v.vendor_location IN ('san francisco', 'san francisco ', ' san francisco ') THEN 'San Francisco'
        WHEN v.vendor_location IN ('san jose', 'san jose ') THEN 'San Jose'
        WHEN v.vendor_location IN ('oakland', 'oakland ') THEN 'Oakland'
        WHEN v.vendor_location IN ('santa clara', 'santa clara ') THEN 'Santa Clara'
        WHEN v.vendor_location IN ('berkeley', 'berkeley ') THEN 'Berkeley' -- Our first case
        WHEN v.vendor_location IN ('hayward', 'hayward ') THEN 'Hayward'
        WHEN v.vendor_location IN ('los gatos', 'los gatos ') THEN 'Los Gatos'
        WHEN v.vendor_location IN ('livermore ', 'livermore') THEN 'Livermore'
        WHEN v.vendor_location IN ('walnut creek ', 'walnut creek') THEN 'Walnut Creek'
        WHEN v.vendor_location IN ('sausolito', 'sausalito ','sausalito') THEN 'Sausalito'
        WHEN v.vendor_location IN ('san anselmo ', 'san alselmo') THEN 'San Alselmo'
        WHEN v.vendor_location IN ('redwood city', 'redwood') THEN 'Redwood'
        WHEN v.vendor_location IN ('fremont', 'freemont ') THEN 'Freemont'
        WHEN v.vendor_location IN ('concord ', 'concord',' concord ') THEN 'Concord'
        ELSE v.vendor_location
    END AS standardized_location,
CASE
        WHEN standardized_location IN ('San Francisco', 'Oakland', 'Berkeley', 'Hayward', 'Livermore', -- Our second Case
                                       'Los Gatos', 'Walnut Creek', 'Concord', 'castro valley', 
                                       'san rafael', 'novato ', 'napa', 'brisbane', 
                                       'san luis obispo', 'San Alselmo', 'sacramento', 'daly city', 
                                       'morro bay', 'monterey', 'milbrae ', 'martinez', 
                                       'greenbrae', 'el cerrito ', 'alameda', 'palo alto', 
                                       'burlingame', 'sunnyvale', 'hillsborough', 'lafayette', 
                                       'san leandro', 'san carlos', 'gilroy', 'san mateo ', 
                                       'south san francisco', 'scotts valley', 'paso robles', 'petaluma', 
                                       'watsonville', 'vacaville ', 'tiburon ', 'sunol ', 'studio', 
                                       'stanford ', 'san ramon ', 'san joaquin valley', 'san diego', 
                                       'saint martin', 'richmond ', 'pleasanton', 'pleasant hill', 
                                       'pittsburg', 'pescadero ', 'oakley ', 'oakley', 'nicasio ', 
                                       'menlo park ', 'mammoth lakes ', 'hollister ', 'hercules',
                                       'felton ', 'felton', 'dublin ', 'dixon ', 'cupertino ', 
                                       'cupertino', 'corte madera', 'cloverdale ', 'clayton ', 
                                       'carmel', 'campbell ', 'campbell', 'calistoga ', 'brentwood ', 
                                       'belmont', 'antioch', 'acampo ') THEN 'San Francisco Bay Area'
        WHEN standardized_location IN ('San Jose', 'Santa Clara') THEN 'South Bay'
        WHEN standardized_location IN ('Sausalito', 'Tiburon') THEN 'North Bay'
        WHEN standardized_location IN ('Fremont', 'Livermore', 'Pleasanton ', 'mountain view ', 'half moon bay ') THEN 'East Bay'
        WHEN standardized_location IN ('palo alto', 'Redwood', 'Menlo Park') THEN 'Peninsula'
        WHEN standardized_location = 'online' THEN 'Online'
        ELSE 'Other'
    END AS region_group,
v.vendor_sustainable, 
CASE -- Our thrird Case
	WHEN v.vendor_rating = 0 THEN ''
    ELSE v.vendor_rating
END AS rating_vendor,
CASE
	WHEN p.unit_vol IN ('per person','1 per bride side','1 per bride trial','1 per groom side','1 per kids','1 per bride','1 per person') THEN 'Individual Services'
    WHEN p.unit_vol IN ('per service','1 per vendor','1','unit') THEN 'Per Unit Services'
    WHEN p.unit_vol IN ('per 100 invitations','6 hours','1 per table','1 per piece','1 per chair','1 per traditional','1 per backdrop','1 per airbrush') THEN 'Event Supplies'
    ELSE 'Others'
    END AS unit_v, -- Our 4th Case
fs.flower_season, fs.flower_style, vs.artificial_flowers_in_portfolio,
att.color, att.tie, att.number_of_buttons, att.lapel_style,
cat.category_name,
sus.equip_ec, sus.avg_usage_hours, sus.total_ec, sus.number_equipment
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id -- Joining all tables
LEFT JOIN Flower_Season_Style as fs
ON p.product_id = fs.product_id
LEFT JOIN Flowers_Vendor_Sustainability as vs
ON v.vendor_id = vs.vendor_id
LEFT JOIN attire as att
ON p.product_id = att.product_id
LEFT JOIN categories as cat
ON p.product_id = cat.product_id
LEFT JOIN Sustainability as sus
ON v.vendor_id = sus.vendor_id
LEFT JOIN ( -- Creating a subquery such as table source
SELECT vendor_id,
CASE
        WHEN v.vendor_location IN ('san francisco', 'san francisco ', ' san francisco ') THEN 'San Francisco'
        WHEN v.vendor_location IN ('san jose', 'san jose ') THEN 'San Jose'
        WHEN v.vendor_location IN ('oakland', 'oakland ') THEN 'Oakland'
        WHEN v.vendor_location IN ('santa clara', 'santa clara ') THEN 'Santa Clara'
        WHEN v.vendor_location IN ('berkeley', 'berkeley ') THEN 'Berkeley'
        WHEN v.vendor_location IN ('hayward', 'hayward ') THEN 'Hayward'
        WHEN v.vendor_location IN ('los gatos', 'los gatos ') THEN 'Los Gatos'
        WHEN v.vendor_location IN ('livermore ', 'livermore') THEN 'Livermore'
        WHEN v.vendor_location IN ('walnut creek ', 'walnut creek') THEN 'Walnut Creek'
        WHEN v.vendor_location IN ('sausolito', 'sausalito ','sausalito') THEN 'Sausalito'
        WHEN v.vendor_location IN ('san anselmo ', 'san alselmo') THEN 'San Alselmo'
        WHEN v.vendor_location IN ('redwood city', 'redwood') THEN 'Redwood'
        WHEN v.vendor_location IN ('fremont', 'freemont ') THEN 'Freemont'
        WHEN v.vendor_location IN ('concord ', 'concord',' concord ') THEN 'Concord'
        ELSE v.vendor_location
    END AS standardized_location
    FROM Vendors as V) AS sl
    ON v.vendor_id = sl.vendor_id; -- This is the table we are going to use for the analysis.
```
## Data Analysis using Python

Python code block:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
data = pd.read_csv('data_final.csv')

# Displaying the first few rows of the dataset
data.head()

#Defining the numerical and categorical attributes - Main tables

df.info(verbose=True)
numerical = ['price_unit','price_ce','rating_vendor']
categ = ['unit_vol','vendor_depart','vendor_location','standardized_location','region_group','unit_v']


#Show the distribution of numerical attributes of Sustainable suppliers 
import numpy as np
import matplotlib.mlab as mlab

plt.style.use('ggplot')
num_bins = 50
df_0 = df[df.vendor_sustainable == 1]

for i in numerical:
    n, bins, patches = plt.hist(df_0[i], num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel(i)
    plt.show()
```

Output:
<img src="{{ site.url }}{{ site.baseurl }}/images/Business-Analysis-Wedding/output_1.jpg" alt="linearly separable data">


```python
#Show the distribution of numerical attributes of Non-Sustainable suppliers 
import numpy as np
import matplotlib.mlab as mlab

plt.style.use('ggplot')
num_bins = 50
df_0 = df[df.vendor_sustainable == 0]

for i in numerical:
    n, bins, patches = plt.hist(df_0[i], num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel(i)
    plt.show()
```
Output:
<img src="{{ site.url }}{{ site.baseurl }}/images/Business-Analysis-Wedding/output_2.jpg" alt="linearly separable data">

```python
#Calculating a Sustainability Ratio per categorical attribute
#Source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
#https://stackoverflow.com/questions/45635539/pandas-concat-with-columns-of-same-categories-turns-to-object

for i in categ:
    brief = pd.concat([df[df.vendor_sustainable == 0].groupby(i).vendor_sustainable.count()
                          , df[df.vendor_sustainable == 1].groupby(i).vendor_sustainable.count()], axis=1)
    brief.columns = ['No_Sus','Sus']
    brief_f = brief.Sus / (brief.Sus + brief.No_Sus)
    plt.figure(figsize=(10,5))
    ax = brief_f.plot(kind = 'bar', color = 'b')
    ax.set_xticklabels(brief_f.index, rotation=30, fontsize=8, ha='right')
    ax.set_xlabel(i)
    ax.set_ylabel('Sustainability Ratio')
```
Output:
<img src="{{ site.url }}{{ site.baseurl }}/images/Business-Analysis-Wedding/output_3.jpg" alt="linearly separable data">

**First Insight**:

We can observe that there is a differentiated distribution in the numerical attributes, specifically in "price_ce," when companies are sustainable compared to when they are not. Furthermore, in the categorical variables, we can see that "catering" and "photo and video" have a high sustainability ratio, and that in East Bay, San Francisco Bay, and Online are the areas with the highest number of sustainable companies.

Let's explore more our data:

```python
# specifying plot size (making it bigger)
fig, ax = plt.subplots( figsize = (12 , 12) )


# developing a freezing cold heatmap
sns.heatmap(data       = df_corr, # the correlation matrix
            cmap       = 'Blues'      , # changing to COOL colors
            square     = True         , # tightening the layout
            annot      = True         , # should there be numbers in the heatmap
            linecolor  = 'black'      , # lines between boxes
            linewidths = 0.5          ) # how thick should the lines be?


# title and displaying the plot
plt.title(label = """
Linear Correlation Heatmap
""")

# rendering the visualization
plt.show(block = True)
```
Output:
<img src="{{ site.url }}{{ site.baseurl }}/images/Business-Analysis-Wedding/output_4.jpg" alt="linearly separable data">

**Second Insight**:

As we can observe, there isn't a significant difference in the rating whether vendors are sustainable or not. The correlation between sustainability and the rating is -0.19, and almost all the data points are above 45 on a scale of 0 to 50. In other words, 90% of the vendors have a rating higher than 45 points, regardless of whether they are sustainable or not.

Let's see some other information about our data.

```python
# Create variables dummy for 'region_group'
#Source: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
#Source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html

dummy_variables = pd.get_dummies(df['region_group'], prefix='Region')
dummy_variables = dummy_variables.astype(int)
data_with_dummies = pd.concat([df, dummy_variables], axis=1)

#We have created dummy variables to convert categorical attributes into binary values. 
#This way, we could perform a Pearson correlation and check if there is a relationship 
#between geographical regions and whether companies are sustainable or not.
data_with_dummies.info()

dummies = ['Region_East Bay', 'Region_North Bay','Region_Online','Region_Other','Region_Peninsula','Region_San Francisco Bay Area','Region_South Bay','vendor_sustainable','price_unit']
data_dummies_corr= data_with_dummies[dummies].corr(method = 'pearson').round(decimals = 2)

# specifying plot size (making it bigger)
fig, ax = plt.subplots( figsize = (12 , 12) )

# developing a freezing cold heatmap
sns.heatmap(data       = data_dummies_corr, # the correlation matrix
            cmap       = 'Blues'      , # changing to COOL colors
            square     = True         , # tightening the layout
            annot      = True         , # should there be numbers in the heatmap
            linecolor  = 'black'      , # lines between boxes
            linewidths = 0.5          ) # how thick should the lines be?

# title and displaying the plot
plt.title(label = """
Linear Correlation Heatmap
""")

# rendering the visualization
plt.show(block = True)
```
Output:
<img src="{{ site.url }}{{ site.baseurl }}/images/Business-Analysis-Wedding/output_5.jpg" alt="linearly separable data">

## Actionable Insights

- **Insight 1**: According to the data obtained, it cannot be stated that there is a linear relationship between sustainability and prices. In the price distribution charts, there is no noticeable change in behavior when filtering for vendors with sustainable practices. Likewise, in the correlation analysis, it was evident that the correlation between price and sustainability is 0.15, which is not strong enough to claim a linear correlation between price and sustainability.<br>
- **Recommendation**: It can be demonstrated that vendors do not increase their prices due to the use of sustainable practices. This may allow vendors to adopt sustainable practices in their businesses as a competitive advantage in the market without being significantly affected by costs. However, according to an essay from ESADE Business School, it is important to evaluate the costs, as there is a tendency for increased costs when implementing sustainable practices.<br>
Source: [Esade](https://dobetter.esade.edu/en/sustainability-price#:~:text=Sustainable%20business%20practices%20lead%20to,not%20to%20purchase%20at%20all)  .

- **Insight 2**: We have observed in the sustainability ratio charts by region that both East Bay, San Francisco Bay Area, and the online channel have a higher proportion of sustainable providers compared to the southern and northern regions.<br>
- **Recommendation**: There is a significant opportunity in the southern and northern regions to implement sustainability practices as a measure to gain a competitive edge in those areas. According to an article by Ernst and Young, implementing sustainability-based strategies is a competitive advantage that sets you apart from the market. Therefore, it is suggested to adopt these practices and transform them into added value.<br>
Source: [EY](https://www.ey.com/en_us/sustainability/sustainability-strategies-create-competitive-advantage)
<br>

## Final Considerations and Answering the Business question

It may be worth considering conducting further in-depth market research in regions where sustainability practices are higher, such as in the San Francisco Bay Area. Additionally, it could be argued that there is a need to raise consumer awareness about the benefits of choosing vendors with sustainable practices.

Addressing the business question: The data does not reflect a direct correlation between price and the sustainability practices of vendors. Both sustainable and non-sustainable vendors offer price ranges for different segments. However, it is noticeable that the implementation of sustainable practices does vary by region.

## Summary

This project provides valuable insights into the cost-effectiveness of sustainable practices in the wedding industry. The SQL queries help in extracting relevant data, and the Python analysis aids in visualizing and understanding the cost dynamics. The findings suggest that while sustainable wedding providers may have a higher average price, the difference is not significant enough to deter cost-conscious consumers.
