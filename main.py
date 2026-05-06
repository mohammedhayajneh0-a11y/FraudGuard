#!/usr/bin/env python
# coding: utf-8

# In[243]:


#Load Data


# In[244]:


import pandas as pd


# In[245]:


fraud_df = pd.read_csv("Fraud_Data.csv")
ip_df = pd.read_csv("IpAddress_to_Country.csv")


# In[246]:


# Inspect dataset structure, sample records, and basic statistics


# In[247]:


print(ip_df.head())
print()
fraud_df.head()


# In[248]:


fraud_df.info()


# In[249]:


ip_df.info()


# In[250]:


fraud_df[["age","purchase_value"]].describe()


# In[251]:


#Data Cleaning


# In[252]:


# Convert signup and purchase timestamps to datetime format
fraud_df["signup_time"]=pd.to_datetime(fraud_df["signup_time"])
fraud_df["purchase_time"]=pd.to_datetime(fraud_df["purchase_time"])


# In[253]:


# Check for duplicate records in both datasets
int(fraud_df.duplicated().sum())


# In[254]:


int(ip_df.duplicated().sum())


# In[255]:


# Check missing values to understand data quality issues
fraud_df.isnull().sum()


# In[256]:


ip_df.isna().sum()


# In[257]:


## Detect outliers using the IQR method


# In[258]:


def check_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(col)
    print("Count:", len(outliers))
    print("Percentage:", f"{len(outliers)/len(df)*100:.2f}%")

    if len(outliers) > 0:
        print("Max:", outliers[col].max())
        print("Min:", outliers[col].min())

check_outliers(fraud_df, "purchase_value")
print("-------------------------------")
check_outliers(fraud_df, "age")


# In[259]:


import matplotlib.pyplot as plt

fraud_df.boxplot(column="purchase_value")
plt.show()

fraud_df.boxplot(column="age")
plt.show()


# In[260]:


#Feature Engineering


# In[261]:


# Create age groups to simplify age-based fraud analysis
import numpy as np
bins=[18,25,35,50,100]
group_names=["18-25","26-35","36-50","50+"]
fraud_df["age_binned"]=pd.cut(
    fraud_df["age"],
    bins=bins,
    labels=group_names,
    include_lowest=True



)

fraud_df[["age", "age_binned"]].head(5)


# In[262]:


# Extract time-based features from signup and purchase timestamps


# In[263]:


fraud_df["signup_hour"]=fraud_df["signup_time"].dt.hour
fraud_df["signup_hour"].head()


# In[264]:


fraud_df["signup_day"]=fraud_df["signup_time"].dt.day
fraud_df["signup_day"].head()


# In[265]:


fraud_df["purchase_hour"]=fraud_df["purchase_time"].dt.hour
fraud_df["purchase_hour"].head()


# In[266]:


fraud_df["purchase_day"]=fraud_df["purchase_time"].dt.day
fraud_df["purchase_day"].head()


# In[267]:


fraud_df["purchase_month"]=fraud_df["purchase_time"].dt.month
fraud_df["purchase_month"].head()


# In[268]:


# Calculate the time difference between signup and purchase
fraud_df["diff_time"]= fraud_df["purchase_time"] -fraud_df["signup_time"]
fraud_df["diff_time"].sort_values(ascending=False).head(10)


# In[269]:


# Convert time difference into hours and days for easier analysis


# In[270]:


fraud_df["diff_day"] = fraud_df["diff_time"].dt.total_seconds() / 86400

fraud_df["diff_day"].sort_values(ascending=False).head(10)


# In[271]:


fraud_df["diff_hours"] = fraud_df["diff_time"].dt.total_seconds() / 3600

fraud_df["diff_hours"].sort_values(ascending=False).head(10)


# In[272]:


# Categorize users based on how quickly they made a purchase after signup
bins=[0,1,3,6,12,24,72,168,float("inf")]

labels=["<1 h","1-3 h","3-6 h","6-12 h","12-24 h","1-3 d","3-7 d",">7 d"]
fraud_df["time_category"]=pd.cut(
    fraud_df["diff_hours"],
    bins=bins,
    labels=labels,
    include_lowest=True

)

fraud_df["time_category"].value_counts()


# In[273]:


# Create Daytime/Nighttime feature based on purchase hour
def time_period(hour):
    if 6 <= hour < 18:
        return "Daytime"
    else:
        return "Nighttime"

fraud_df["time_of_day"] = fraud_df["purchase_hour"].apply(time_period)
fraud_df["time_of_day"].head()


# In[274]:


# Behavior Analysis


# In[275]:


# Analyze overall fraud distribution
print(fraud_df["class"].value_counts())
fraud_df["class"].value_counts(normalize=True)


# In[276]:


#Univariate Analysis


# In[277]:


# Compare purchase value across fraud and non-fraud transactions


# In[278]:


fraud_df.groupby(["class","age_binned"])["purchase_value"].mean()


# In[279]:


fraud_df.groupby(["class","source"])["purchase_value"].mean()


# In[280]:


fraud_df.groupby(["class","browser"])["purchase_value"].mean()


# In[281]:


fraud_df.groupby(["class","sex"])["purchase_value"].mean()


# In[282]:


# Analyze fraud rate by browser, source, sex, and age group


# In[283]:


pd.crosstab(fraud_df["browser"], fraud_df["class"], normalize="index").sort_values(by=1, ascending=False)


# In[284]:


pd.crosstab(fraud_df["sex"], fraud_df["class"],normalize="index").sort_values(by=1,ascending=False)


# In[285]:


pd.crosstab(fraud_df["age_binned"], fraud_df["class"],normalize="index").sort_values(by=1,ascending=False)


# In[286]:


pd.crosstab(fraud_df["source"], fraud_df["class"],normalize="index" ).sort_values(by=1,ascending=False)


# In[287]:


pd.crosstab(
    fraud_df["time_category"],
    fraud_df["class"],
    normalize="index"
).sort_values(by=1, ascending=False)


# In[288]:


# fraud vs time diff
fraud_df.groupby("class")["diff_hours"].agg(["mean","count"])


# In[289]:


pd.crosstab(
    [fraud_df["time_category"], fraud_df["time_of_day"]],
    fraud_df["class"],
    normalize="index"
)


# In[290]:


#Combination Analysis


# In[291]:


source_browser_time_analysis=fraud_df.groupby(["source","browser","time_category"]).agg(fraud_rate=("class","mean"),count=("class","count")).reset_index()

source_browser_time_analysis.sort_values(by="fraud_rate", ascending=False).head()


# In[292]:


demographic_time_analysis=fraud_df.groupby(["sex","age_binned","time_category"]).agg(fraud_rate=("class","mean"),count=("class","count")).reset_index()

demographic_time_analysis.sort_values(by="fraud_rate", ascending=False).head()


# In[293]:


#Country Analysis


# In[294]:


# Sort datasets before performing range-based IP matching
fraud_df = fraud_df.sort_values("ip_address")
ip_df = ip_df.sort_values("lower_bound_ip_address")


# In[295]:


# Match each IP address to its country using range-based merge
# Use merge_asof to match each transaction IP with the closest lower IP range,
# then validate that the IP is still within the upper bound.
merged = pd.merge_asof(
    fraud_df,
    ip_df,
    left_on="ip_address",
    right_on="lower_bound_ip_address",
    direction="backward"
)


# In[296]:


merged = merged[
    merged["ip_address"] <= merged["upper_bound_ip_address"]
]

merged.head()


# In[297]:


merged["country"].value_counts(normalize=True).head()


# In[298]:


# country time fraud rate
purchase_by_group = merged.groupby(
    ["class","country","time_category"]
)["purchase_value"].agg(["mean","count"]).reset_index()
# filter low counts
purchase_by_group = purchase_by_group[purchase_by_group["count"] > 100]
# top fraud countries

purchase_by_group.sort_values(by=["mean","count"], ascending=[False, False]).head(10)


# In[299]:


# purchase behavior analysis
fraud_by_country_time = merged.groupby(
    ["country", "time_category"]
).agg(
    fraud_rate=("class", "mean"),
    count=("class", "count"),
    avg_purchase=("purchase_value", "mean")
).reset_index()
# filter low counts
fraud_by_country_time = fraud_by_country_time[fraud_by_country_time["count"] > 100]

fraud_by_country_time.sort_values(by=["fraud_rate","count"], ascending=[False, False]).head(10)


# In[300]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[301]:


time_fraud = fraud_df.groupby("time_category")["class"].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=time_fraud, x="time_category", y="class")
plt.title("Fraud Rate by Time Category")
plt.xlabel("Time Between Signup and Purchase")
plt.ylabel("Fraud Rate")
plt.xticks(rotation=45)
plt.show()


# In[302]:


avg_time = fraud_df.groupby("class")["diff_hours"].mean().reset_index()

plt.figure(figsize=(6,4))
sns.barplot(data=avg_time, x="class", y="diff_hours")
plt.title("Average Time Difference by Class")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Average Hours")
plt.show()


# In[323]:


age_fraud = fraud_df.groupby("age_binned")["class"].mean().reset_index(name="fraud_rate")



plt.figure(figsize=(7,4))
sns.barplot(data=age_fraud, x="age_binned", y="fraud_rate")
plt.title("Fraud Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Fraud Rate")
plt.show()


# In[325]:


country_fraud = merged.groupby("country").agg(
    fraud_rate=("class","mean"),
    count=("class","count")
).reset_index()

country_fraud = country_fraud[country_fraud["count"] > 100]
country_fraud = country_fraud.sort_values(by="fraud_rate", ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(data=country_fraud, x="country", y="fraud_rate")
plt.title("Top Countries by Fraud Rate")
plt.xlabel("Country")
plt.ylabel("Fraud Rate")
plt.xticks(rotation=45)
plt.show()


# In[307]:


# Heatmap محسنة
pivot = fraud_by_country_time.pivot(index="country", columns="time_category", values="fraud_rate")

# نأخذ فقط الدول اللي عندها بيانات كافية
min_transactions = 100
valid_countries = merged['country'].value_counts()[merged['country'].value_counts() > min_transactions].index
pivot = pivot.loc[pivot.index.isin(valid_countries)]

# نرتب حسب متوسط الـ fraud rate
pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

plt.figure(figsize=(14, 10))
sns.heatmap(pivot, 
            cmap="YlOrRd", 
            annot=True, 
            fmt=".3f",
            linewidths=0.5,
            cbar_kws={'label': 'Fraud Rate'},
            vmin=0, 
            vmax=0.4)   # مهم: حدد vmax حسب البيانات

plt.title("Fraud Rate by Country and Time Category\n(Countries with >100 transactions)", fontsize=14)
plt.tight_layout()
plt.show()


# In[ ]:


pv_time_class = fraud_df.groupby(["time_category","class"])["purchase_value"].mean().reset_index()
plt.figure(figsize=(8,5))
sns.barplot(data=pv_time_class, x="time_category", y="purchase_value", hue="class")
plt.title("Purchase Value by Time Category and Class")
plt.xlabel("Time Category")
plt.ylabel("Average Purchase Value")
plt.xticks(rotation=45)
plt.legend(title="Class")
plt.show()


# In[ ]:


# Fraud Detection Analysis – Key Insights

## 1. Primary Finding
"""
The most significant indicator of fraud is the time between account signup and first purchase.  
Transactions completed within the first hour after signup show a dramatically higher fraud rate.  
As the time interval increases, the probability of fraud decreases substantially."""

## 2. Purchase Value
"""There is no meaningful relationship between purchase value and fraud.  
Fraudulent transactions closely resemble normal transactions in terms of spending behavior.  
This indicates that fraudsters intentionally mimic legitimate users, making this feature weak for prediction."""

## 3. Geographic Patterns
"""Fraud rates vary across countries.  
However, country alone is not a strong indicator.  

The highest risk emerges when combining:
Country + Fast Transaction (<1 hour)

This combination identifies the most critical fraud segments."""

## 4. Behavioral Patterns
"""Fraud in this dataset is primarily behavior-driven rather than attribute-driven.

- Transaction speed is the dominant signal  
- Demographics (age, gender) have limited impact  
- Browser and traffic source show only minor differences  """

## Executive Summary

"""The dominant fraud pattern is:

Account signup followed by an immediate purchase (within 1 hour).

Time-based features are the strongest predictors of fraud, significantly outperforming all other variables.  
In contrast, purchase value and user attributes provide minimal discriminatory power.
"""
## Recommendations
"""
- Implement real-time monitoring for transactions occurring within the first hour  
- Add verification steps (e.g., OTP, 2FA) for fast purchases  
- Prioritize high-risk country + fast transaction combinations  
- Avoid relying on weak signals such as purchase value alone  """

